# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import json
import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, get_token_level_logits

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self._has_dummy_forwarded = False

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        # 添加GRPO相关参数  
        self.use_reasoning_quality = finetuning_args.use_reasoning_quality  
        self.reasoning_weight = finetuning_args.reasoning_weight 

        # 增加KL散度——添加动态β相关属性  
        self.use_dynamic_beta = getattr(finetuning_args, "use_dynamic_beta", False)  
        self.initial_beta = self.beta  # 保存初始β值  
        self.min_beta = getattr(finetuning_args, "min_beta", 0.01)  
        self.beta_decay_strategy = getattr(finetuning_args, "beta_decay_strategy", "linear")  
        self.token_regularization_type = getattr(finetuning_args, "token_regularization_type", "entropy")  
        self.current_epoch = 0  
        self.total_epochs = kwargs.get("args").num_train_epochs  
        self.initial_token_logits = None  # 用于KL散度近似

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        # 初始化ReasoningQualityEvaluator  
        self.reasoning_evaluator = None  
        if self.use_reasoning_quality:  
            from ...eval.reasoning_quality import ReasoningQualityEvaluator  
            self.reasoning_evaluator = ReasoningQualityEvaluator(  
                tokenizer=kwargs.get("tokenizer", None),  
                format_weight=finetuning_args.format_weight,  
                cosine_weight=finetuning_args.cosine_weight,  
                cosine_min_len_value_wrong=finetuning_args.cosine_min_len_value_wrong,  
                cosine_max_len_value_wrong=finetuning_args.cosine_max_len_value_wrong,  
                cosine_min_len_value_correct=finetuning_args.cosine_min_len_value_correct,  
                cosine_max_len_value_correct=finetuning_args.cosine_max_len_value_correct,  
                cosine_max_len=finetuning_args.cosine_max_len,  
                repetition_weight=0.2,  # 可以添加到finetuning_args中  
                overlong_weight=0.2,    # 可以添加到finetuning_args中  
                soft_max_length=2048,   # 可以添加到finetuning_args中  
                soft_cache_length=512   # 可以添加到finetuning_args中  
            )  


        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        r"""
        Replaces the method of KTO Trainer with the one of the standard Trainer.
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss
    
    # +++++++++++原始的SimPo算法++++++++++++++
    # def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
    #     r"""
    #     Computes SimPO loss for batched log probabilities of the policy model.
    #     """
    #     pi_logratios = chosen_logps - rejected_logps
    #     gamma_logratios = self.simpo_gamma / self.beta
    #     logits = pi_logratios - gamma_logratios
    #     simpo_loss = -F.logsigmoid(self.beta * logits)
    #     return simpo_loss
    
    # +++++++++++改进的SimPo算法++++++++++++++
    def simpo_loss(  
        self,   
        chosen_logps: "torch.Tensor",   
        rejected_logps: "torch.Tensor",   
        token_regularization_term: Optional["torch.Tensor"] = None  
    ) -> "torch.Tensor":  
        r"""  
        计算SimPO损失，并支持token级别正则化。  
        """  
        pi_logratios = chosen_logps - rejected_logps  
        gamma_logratios = self.simpo_gamma / self.beta  
        logits = pi_logratios - gamma_logratios  
        simpo_loss = -F.logsigmoid(self.beta * logits)  
        
        # 如果提供了token级别正则化项且启用了动态β  
        if token_regularization_term is not None and self.use_dynamic_beta:  
            # 使用当前β值加权正则化项  
            total_loss = simpo_loss + self.beta * token_regularization_term  
            return total_loss  
            
        return simpo_loss

    # +++++++++新增β值动态更新技术+++++++++++
    def update_dynamic_beta(self):  
        """  
        根据当前训练进度和选择的策略更新β值  
        """  
        if not self.use_dynamic_beta:  
            return  
            
        progress = self.current_epoch / self.total_epochs  
        
        # 根据选择的衰减策略更新β值  
        if self.beta_decay_strategy == "linear":  
            # 线性衰减  
            self.beta = self.min_beta + (self.initial_beta - self.min_beta) * (1 - progress)  
        elif self.beta_decay_strategy == "exponential":  
            # 指数衰减  
            self.beta = self.min_beta + (self.initial_beta - self.min_beta) * math.exp(-5 * progress)  
        elif self.beta_decay_strategy == "cosine":  
            # 余弦衰减  
            self.beta = self.min_beta + 0.5 * (self.initial_beta - self.min_beta) * (1 + math.cos(math.pi * progress))  
        else:  
            # 默认使用线性衰减  
            self.beta = self.min_beta + (self.initial_beta - self.min_beta) * (1 - progress)  
        
        # 记录当前β值  
        self.log({"dynamic_beta": self.beta})

    # +++++++++新增计算正则化方法+++++++++++
    def compute_token_level_regularization(  
        self,   
        token_logits: "torch.Tensor",   
        mask: "torch.Tensor",  
        initial_token_logits: Optional["torch.Tensor"] = None  
    ) -> "torch.Tensor":  
        """  
        计算token级别的正则化项  
        
        Args:  
            token_logits: 形状为[batch_size, seq_len, vocab_size]的logits  
            mask: 形状为[batch_size, seq_len]的mask，表示非padding位置  
            initial_token_logits: 用于KL散度近似的初始logits  
            
        Returns:  
            正则化项，标量值  
        """  
        if self.token_regularization_type == "entropy":  
            # 1. 熵正则化 - 鼓励每个位置上的分布更加多样化  
            token_probs = F.softmax(token_logits, dim=-1)  
            token_logprobs = F.log_softmax(token_logits, dim=-1)  
            # 计算每个位置的熵  
            token_entropy = -torch.sum(token_probs * token_logprobs, dim=-1)  # [batch_size, seq_len]  
            # 只考虑非padding位置，计算平均熵  
            mean_entropy = (token_entropy * mask).sum() / (mask.sum() + 1e-8)  
            # 返回负熵作为正则化项（最大化熵）  
            return -mean_entropy  
            
        elif self.token_regularization_type == "l2":  
            # 2. L2正则化 - 防止logits值过大  
            # 对logits应用softmax得到概率分布  
            token_probs = F.softmax(token_logits, dim=-1)  
            # 计算L2范数  
            l2_norm = torch.sum(token_probs ** 2, dim=-1)  # [batch_size, seq_len]  
            # 只考虑非padding位置，计算平均L2范数  
            mean_l2_norm = (l2_norm * mask).sum() / (mask.sum() + 1e-8)  
            return mean_l2_norm  
            
        elif self.token_regularization_type == "kl_approx" and initial_token_logits is not None:  
            # 3. KL散度近似 - 保持分布接近初始分布  
            # 计算当前分布和初始分布  
            current_probs = F.softmax(token_logits, dim=-1)  
            initial_probs = F.softmax(initial_token_logits, dim=-1)  
            current_logprobs = F.log_softmax(token_logits, dim=-1)  
            initial_logprobs = F.log_softmax(initial_token_logits, dim=-1)  
            
            # 计算KL散度: KL(current||initial)  
            kl_div = torch.sum(current_probs * (current_logprobs - initial_logprobs), dim=-1)  # [batch_size, seq_len]  
            # 只考虑非padding位置，计算平均KL散度  
            mean_kl_div = (kl_div * mask).sum() / (mask.sum() + 1e-8)  
            return mean_kl_div  
        
        elif self.token_regularization_type == "chi_square" and initial_token_logits is not None:  
            # 4. χ²散度 - 对分布尾部更敏感，更有效地约束极端偏离  
            # 计算当前分布和初始分布  
            current_probs = F.softmax(token_logits, dim=-1)  
            initial_probs = F.softmax(initial_token_logits, dim=-1)  
            
            # 添加小的epsilon值防止除零错误  
            epsilon = 1e-10  
            initial_probs = initial_probs + epsilon  
            
            # 计算χ²散度: D_χ²(p||q) = ∑ (p_i - q_i)²/q_i  
            # 其中p是当前分布，q是初始分布  
            chi_square_div = torch.sum(  
                (current_probs - initial_probs)**2 / initial_probs,  
                dim=-1  
            )  # [batch_size, seq_len]  
            
            # 只考虑非padding位置，计算平均χ²散度  
            mean_chi_square_div = (chi_square_div * mask).sum() / (mask.sum() + 1e-8)  
            return mean_chi_square_div  
        
        else:  
            # 默认不使用正则化  
            return torch.tensor(0.0, device=token_logits.device)

    def nca_pair(self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
        losses = (
            -F.logsigmoid(chosen_rewards)
            - 0.5 * F.logsigmoid(-chosen_rewards)
            - 0.5 * F.logsigmoid(-rejected_rewards)
        )
        return losses, chosen_rewards, rejected_rewards
    
    def grpo_loss(  
        self,  
        policy_chosen_logps: torch.FloatTensor,  
        policy_rejected_logps: torch.FloatTensor,  
        reference_chosen_logps: torch.FloatTensor,  
        reference_rejected_logps: torch.FloatTensor,  
        batch: Dict[str, torch.Tensor] = None  
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:  
        """  
        计算GRPO损失函数，结合DPO和推理质量评估。  
        """  
        # 计算基础DPO奖励  
        chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta  
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta  
        
        # 如果启用了推理质量评估  
        if self.use_reasoning_quality and self.reasoning_evaluator is not None and batch is not None:  
            try:  
                # 获取批次大小  
                batch_size = len(policy_chosen_logps)  
                
                # 调试信息：打印批次键  
                if self.is_world_process_zero() and not hasattr(self, "_debug_printed_keys"):  
                    print(f"批次键: {list(batch.keys())}")  
                    self._debug_printed_keys = True  
                
                # 从批次中提取标签  
                if "labels" in batch:  
                    all_labels = batch["labels"]  
                    # 在DPO中，批次结构是前半部分包含chosen样本，后半部分包含rejected样本  
                    chosen_labels = all_labels[:batch_size]  
                    rejected_labels = all_labels[batch_size:]  
                    
                    # 处理特殊标记（如-100）  
                    # 将-100等特殊值替换为tokenizer.pad_token_id  
                    chosen_labels_processed = chosen_labels.clone()  
                    rejected_labels_processed = rejected_labels.clone()  
                    
                    # 替换特殊值（通常是-100）为pad_token_id  
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0  
                    chosen_labels_processed[chosen_labels_processed == IGNORE_INDEX] = pad_token_id  
                    rejected_labels_processed[rejected_labels_processed == IGNORE_INDEX] = pad_token_id  
                    
                    # 解码为文本  
                    chosen_texts = self.tokenizer.batch_decode(chosen_labels_processed, skip_special_tokens=True)  
                    rejected_texts = self.tokenizer.batch_decode(rejected_labels_processed, skip_special_tokens=True)  
                    
                    # 获取ground_truths（如果有的话）  
                    ground_truths = batch.get("ground_truths", None)  
                    
                    # 评估推理质量  
                    chosen_quality, rejected_quality = self.reasoning_evaluator(  
                        chosen_texts, rejected_texts, ground_truths  
                    )  
                    
                    # 将质量分数移动到正确的设备上  
                    chosen_quality = chosen_quality.to(chosen_rewards.device)  
                    rejected_quality = rejected_quality.to(rejected_rewards.device)  
                    
                    # 将推理质量分数添加到奖励中  
                    chosen_rewards += self.reasoning_weight * chosen_quality  
                    rejected_rewards += self.reasoning_weight * rejected_quality  
                else:  
                    if self.is_world_process_zero():  
                        print("警告: 批次中未找到'labels'键。跳过推理质量评估。")  
                        print(f"可用的键: {list(batch.keys())}")  
            except Exception as e:  
                if self.is_world_process_zero():  
                    print(f"推理质量评估中出错: {e}")  
                    import traceback  
                    traceback.print_exc()  
        
        # 计算损失  
        losses = -F.logsigmoid(chosen_rewards - rejected_rewards)  
        
        return losses, chosen_rewards, rejected_rewards

    # +++++++++++++原始的compute_preference_loss函数+++++++++++++
    # 修改这个地方以支持GRPO
    # def compute_preference_loss(
    #     self,
    #     policy_chosen_logps: "torch.Tensor",
    #     policy_rejected_logps: "torch.Tensor",
    #     reference_chosen_logps: Optional["torch.Tensor"],
    #     reference_rejected_logps: Optional["torch.Tensor"],
    #     batch: Optional[Dict[str, "torch.Tensor"]] = None
    # ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    #     r"""
    #     Computes loss for preference learning.
    #     """
    #     if not self.finetuning_args.use_ref_model:
    #         if self.loss_type == "orpo":
    #             losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
    #         elif self.loss_type == "simpo":
    #             losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
    #         elif self.loss_type == "grpo" and self.use_reasoning_quality:  
    #             # 对于不使用参考模型的情况，简化GRPO实现  
    #             chosen_rewards = self.beta * policy_chosen_logps  
    #             rejected_rewards = self.beta * policy_rejected_logps  
                
    #             # 如果启用了推理质量评估  
    #             if self.reasoning_evaluator is not None and batch is not None:  
    #                 # 获取文本内容  
    #                 batch_size = len(policy_chosen_logps)  
    #                 chosen_texts = self.tokenizer.batch_decode(  
    #                     batch["chosen_labels"], skip_special_tokens=True  
    #                 )  
    #                 rejected_texts = self.tokenizer.batch_decode(  
    #                     batch["rejected_labels"], skip_special_tokens=True  
    #                 )  
                    
    #                 # 评估推理质量  
    #                 chosen_quality, rejected_quality = self.reasoning_evaluator(  
    #                     chosen_texts, rejected_texts, None  
    #                 )  
                    
    #                 # 将质量评分转移到正确的设备上  
    #                 chosen_quality = chosen_quality.to(chosen_rewards.device)  
    #                 rejected_quality = rejected_quality.to(rejected_rewards.device)  
                    
    #                 # 将推理质量评分添加到奖励中  
    #                 chosen_rewards += self.reasoning_weight * chosen_quality  
    #                 rejected_rewards += self.reasoning_weight * rejected_quality  
                
    #             losses = -F.logsigmoid(chosen_rewards - rejected_rewards)  
    #         else:  
    #             raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")  
    
    #         chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()  
    #         rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()  
    #     else:
    #         if self.loss_type == "nca_pair":
    #             losses, chosen_rewards, rejected_rewards = self.nca_pair(
    #                 policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
    #             )
    #         elif self.loss_type == "grpo" and self.use_reasoning_quality:  
    #             losses, chosen_rewards, rejected_rewards = self.grpo_loss(  
    #                 policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, batch  
    #             )  
    #         else:
    #             losses, chosen_rewards, rejected_rewards = self.dpo_loss(
    #                 policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
    #             )

    #     return losses, chosen_rewards, rejected_rewards

    # +++++++++++++修改后的compute_preference_loss++++++++++++
    def compute_preference_loss(  
        self,  
        policy_chosen_logps: "torch.Tensor",  
        policy_rejected_logps: "torch.Tensor",  
        reference_chosen_logps: Optional["torch.Tensor"],  
        reference_rejected_logps: Optional["torch.Tensor"],  
        batch: Optional[Dict[str, "torch.Tensor"]] = None,  
        policy_chosen_token_logits: Optional["torch.Tensor"] = None,  
        policy_chosen_token_mask: Optional["torch.Tensor"] = None,  
        reference_chosen_token_logits: Optional["torch.Tensor"] = None,  
        reference_chosen_token_mask: Optional["torch.Tensor"] = None,  
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:  
        r"""  
        计算偏好学习的损失，支持token级别正则化。  
        """  
        # 更新动态β（如果启用）  
        if self.use_dynamic_beta:  
            self.update_dynamic_beta()  
            
        # 计算token级别正则化项  
        token_regularization_term = None  
        if self.use_dynamic_beta and policy_chosen_token_logits is not None and policy_chosen_token_mask is not None:  
            # 如果是KL散度近似且有参考模型  
            if self.token_regularization_type == "kl_approx" and reference_chosen_token_logits is not None:  
                token_regularization_term = self.compute_token_level_regularization(  
                    policy_chosen_token_logits,   
                    policy_chosen_token_mask,  
                    reference_chosen_token_logits  
                )  
            else:  
                # 对于熵正则化和L2正则化，或者没有参考模型的KL散度近似  
                if self.token_regularization_type in ["kl_approx", "chi_square"] and self.initial_token_logits is None:  
                    # 首次调用时保存当前logits作为初始状态  
                    with torch.no_grad():  
                        self.initial_token_logits = policy_chosen_token_logits.detach().clone() 
                    
                token_regularization_term = self.compute_token_level_regularization(  
                    policy_chosen_token_logits,   
                    policy_chosen_token_mask,  
                    self.initial_token_logits if self.token_regularization_type == "kl_approx" else None  
                )  
        
        if not self.finetuning_args.use_ref_model:  
            if self.loss_type == "orpo":  
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)  
                # 如果启用了动态β和正则化，也为orpo添加正则化项  
                if token_regularization_term is not None and self.use_dynamic_beta:  
                    losses = losses + self.beta * token_regularization_term  
            elif self.loss_type == "simpo":  
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps, token_regularization_term)  
            elif self.loss_type == "grpo" and self.use_reasoning_quality:    
                # 对于GRPO，可以类似地修改  
                # 先计算基本的GRPO损失  
                chosen_rewards = self.beta * policy_chosen_logps    
                rejected_rewards = self.beta * policy_rejected_logps    
                
                # 如果启用了推理质量评估    
                if self.reasoning_evaluator is not None and batch is not None:    
                    # 获取文本内容    
                    batch_size = len(policy_chosen_logps)    
                    chosen_texts = self.tokenizer.batch_decode(    
                        batch["chosen_labels"], skip_special_tokens=True    
                    )    
                    rejected_texts = self.tokenizer.batch_decode(    
                        batch["rejected_labels"], skip_special_tokens=True    
                    )    
                    
                    # 评估推理质量    
                    chosen_quality, rejected_quality = self.reasoning_evaluator(    
                        chosen_texts, rejected_texts, None    
                    )    
                    
                    # 将质量评分转移到正确的设备上    
                    chosen_quality = chosen_quality.to(chosen_rewards.device)    
                    rejected_quality = rejected_quality.to(rejected_rewards.device)    
                    
                    # 将推理质量评分添加到奖励中    
                    chosen_rewards += self.reasoning_weight * chosen_quality    
                    rejected_rewards += self.reasoning_weight * rejected_quality    
                
                losses = -F.logsigmoid(chosen_rewards - rejected_rewards)  
                
                # 添加正则化项  
                if token_regularization_term is not None and self.use_dynamic_beta:  
                    losses = losses + self.beta * token_regularization_term  
            else:  
                # 对于其他损失类型，如果启用了动态β，也添加正则化  
                base_losses, chosen_rewards, rejected_rewards = self.dpo_loss(  
                    policy_chosen_logps, policy_rejected_logps, None, None  
                )  
                if token_regularization_term is not None and self.use_dynamic_beta:  
                    losses = base_losses + self.beta * token_regularization_term  
                else:  
                    losses = base_losses  
            
            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()  
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()  
        else:  
            # 使用参考模型的情况...  
            if self.loss_type == "nca_pair":  
                losses, chosen_rewards, rejected_rewards = self.nca_pair(  
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps  
                )  
            elif self.loss_type == "grpo" and self.use_reasoning_quality:    
                losses, chosen_rewards, rejected_rewards = self.grpo_loss(    
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, batch    
                )  
                
                # 添加正则化项  
                if token_regularization_term is not None and self.use_dynamic_beta:  
                    losses = losses + self.beta * token_regularization_term  
            else:  
                losses, chosen_rewards, rejected_rewards = self.dpo_loss(  
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps  
                )  
                
                # 即使使用参考模型，如果启用了动态β和正则化，也添加正则化项  
                if token_regularization_term is not None and self.use_dynamic_beta:  
                    losses = losses + self.beta * token_regularization_term  
    
        return losses, chosen_rewards, rejected_rewards

    # ++++++++++++++可选方案1++++++++++++++++
    def adjust_beta_by_performance(self, eval_loss):  
        """  
        根据验证集性能动态调整β值  
        """  
        if not self.use_dynamic_beta:  
            return  
            
        if not hasattr(self, "prev_eval_loss"):  
            self.prev_eval_loss = eval_loss  
        else:  
            # 如果性能改善，减小β；否则增大β  
            if eval_loss < self.prev_eval_loss:  
                self.beta = max(self.min_beta, self.beta * 0.9)  
            else:  
                self.beta = min(self.initial_beta, self.beta * 1.1)  
            self.prev_eval_loss = eval_loss  
            
            # 记录调整后的β值  
            self.log({"dynamic_beta_adjusted": self.beta})

    # ++++++++++++++可选方案2++++++++++++++++
    def evaluate(self, *args, **kwargs):  
        metrics = super().evaluate(*args, **kwargs)  
        
        # 如果启用了动态β，根据验证性能调整β值  
        if self.use_dynamic_beta and "eval_loss" in metrics:  
            self.adjust_beta_by_performance(metrics["eval_loss"])  
            
        return metrics 
    
    def on_epoch_end(self, args, state, control, **kwargs):  
        """  
        在每个epoch结束时更新计数  
        """  
        if self.use_dynamic_beta:  
            self.current_epoch += 1  
            # 更新β值  
            self.update_dynamic_beta()

    #  ++++++++++++原始的concatenated_forward实现++++++++++++++
    # @override
    # def concatenated_forward(
    #     self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    # ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    #     r"""
    #     Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

    #     Otherwise the average log probabilities.
    #     """
    #     if self.finetuning_args.use_ref_model:
    #         batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

    #     all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
    #     all_logps, valid_length = get_batch_logps(
    #         logits=all_logits, labels=batch["labels"], shift_labels=model.sequence_parallel_group is None
    #     )  # shift labels if no sequence parallel
    #     if self.loss_type in ["ipo", "orpo", "simpo"]:
    #         all_logps = all_logps / valid_length

    #     batch_size = batch["input_ids"].size(0) // 2
    #     chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
    #     chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
    #     chosen_length, _ = valid_length.split(batch_size, dim=0)
    #     return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_length


    @override  
    def concatenated_forward(  
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]  
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:  
        r"""  
        计算模型的前向传播，返回序列级别的对数概率和token级别的logits。  
        """  
        if self.finetuning_args.use_ref_model:  
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error  
    
        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)  
        all_logps, valid_length = get_batch_logps(  
            logits=all_logits, labels=batch["labels"], shift_labels=model.sequence_parallel_group is None  
        )  # shift labels if no sequence parallel  
        
        # 获取token级别的logits和mask  
        token_logits, token_mask = get_token_level_logits(  
            logits=all_logits, labels=batch["labels"], shift_labels=model.sequence_parallel_group is None  
        )  
        
        if self.loss_type in ["ipo", "orpo", "simpo"]:  
            all_logps = all_logps / valid_length  
    
        batch_size = batch["input_ids"].size(0) // 2  
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)  
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)  
        chosen_length, _ = valid_length.split(batch_size, dim=0)  
        
        # 分割token级别的logits和mask  
        chosen_token_logits, rejected_token_logits = token_logits.split(batch_size, dim=0)  
        chosen_token_mask, rejected_token_mask = token_mask.split(batch_size, dim=0)  
        
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_length, chosen_token_logits, chosen_token_mask

    # +++++++++++原始的compute_reference_log_probs实现+++++++++++
    # @override
    # def compute_reference_log_probs(
    #     self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    # ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
    #     r"""
    #     Computes log probabilities of the reference model.
    #     """
    #     if not self.finetuning_args.use_ref_model:
    #         return None, None

    #     if self.ref_model is None:
    #         ref_model = model
    #         ref_context = self.accelerator.unwrap_model(model).disable_adapter()
    #     else:
    #         ref_model = self.ref_model
    #         ref_context = nullcontext()

    #     with torch.no_grad(), ref_context:
    #         reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

    #     return reference_chosen_logps, reference_rejected_logps

    @override  
    def compute_reference_log_probs(  
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]  
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"], Optional["torch.Tensor"], Optional["torch.Tensor"]]:  
        r"""  
        计算参考模型的对数概率和token级别logits。  
        """  
        if not self.finetuning_args.use_ref_model:  
            return None, None, None, None  
    
        if self.ref_model is None:  
            ref_model = model  
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()  
        else:  
            ref_model = self.ref_model  
            ref_context = nullcontext()  
    
        with torch.no_grad(), ref_context:  
            reference_chosen_logps, reference_rejected_logps, _, _, _, reference_chosen_token_logits, reference_chosen_token_mask = self.concatenated_forward(ref_model, batch)  
    
        return reference_chosen_logps, reference_rejected_logps, reference_chosen_token_logits, reference_chosen_token_mask

    @override
    def get_batch_loss_metrics(  
        self,  
        model: "PreTrainedModel",  
        batch: Dict[str, "torch.Tensor"],  
        train_eval: Literal["train", "eval"] = "train",  
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:  
        """  
        计算DPO损失和其他指标。  
        """  
        metrics = {}  
        (  
            policy_chosen_logps,  
            policy_rejected_logps,  
            policy_chosen_logits,  
            policy_rejected_logits,  
            policy_chosen_length,  
        ) = self.concatenated_forward(model, batch)  
    
        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)  
    
        # 处理序列并行  
        sp_group = model.sequence_parallel_group  
        if sp_group is not None:  
            policy_chosen_logps = dist.nn.all_reduce(policy_chosen_logps, op=dist.ReduceOp.SUM, group=sp_group)  
            policy_rejected_logps = dist.nn.all_reduce(policy_rejected_logps, op=dist.ReduceOp.SUM, group=sp_group)  
            reference_chosen_logps = dist.nn.all_reduce(reference_chosen_logps, op=dist.ReduceOp.SUM, group=sp_group)  
            reference_rejected_logps = dist.nn.all_reduce(reference_rejected_logps, op=dist.ReduceOp.SUM, group=sp_group)  
            policy_chosen_length = dist.nn.all_reduce(policy_chosen_length, op=dist.ReduceOp.SUM, group=sp_group)  
    
        # 存储推理质量评估结果  
        chosen_quality = None  
        rejected_quality = None  
        
        # 如果启用了推理质量评估，计算并存储结果  
        if self.use_reasoning_quality and self.reasoning_evaluator is not None and batch is not None:  
            try:  
                # 获取批次大小  
                batch_size = len(policy_chosen_logps)  
                
                # 从批次中提取标签  
                if "labels" in batch:  
                    all_labels = batch["labels"]  
                    chosen_labels = all_labels[:batch_size]  
                    rejected_labels = all_labels[batch_size:]  
                    
                    # 处理特殊标记（如-100）  
                    chosen_labels_processed = chosen_labels.clone()  
                    rejected_labels_processed = rejected_labels.clone()  
                    
                    # 替换特殊值为pad_token_id  
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0  
                    chosen_labels_processed[chosen_labels_processed == IGNORE_INDEX] = pad_token_id  
                    rejected_labels_processed[rejected_labels_processed == IGNORE_INDEX] = pad_token_id  
                    
                    # 解码为文本  
                    chosen_texts = self.tokenizer.batch_decode(chosen_labels_processed, skip_special_tokens=True)  
                    rejected_texts = self.tokenizer.batch_decode(rejected_labels_processed, skip_special_tokens=True)  
                    
                    # 评估推理质量  
                    chosen_quality, rejected_quality = self.reasoning_evaluator(  
                        chosen_texts, rejected_texts, None  
                    )  
                    
                    # 将质量分数移动到正确的设备上  
                    chosen_quality = chosen_quality.to(policy_chosen_logps.device)  
                    rejected_quality = rejected_quality.to(policy_chosen_logps.device)  
            except Exception as e:  
                if self.is_world_process_zero():  
                    print(f"推理质量评估中出错: {e}")  
                    import traceback  
                    traceback.print_exc()  
    
        # 传递batch以便访问文本内容  
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(  
            policy_chosen_logps,  
            policy_rejected_logps,  
            reference_chosen_logps,  
            reference_rejected_logps,  
            batch  
        )  
    
        # 其余代码保持不变...  
        policy_chosen_logps_avg = policy_chosen_logps / policy_chosen_length  
        sft_loss = -policy_chosen_logps_avg  
        if self.ftx_gamma > 1e-6:  
            losses += self.ftx_gamma * sft_loss  
    
        prefix = "eval_" if train_eval == "eval" else ""  
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()  
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()  
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()  
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()  
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()  
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()  
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()  
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()  
        
        # 添加推理质量评估指标  
        if chosen_quality is not None and rejected_quality is not None:  
            metrics[f"{prefix}grpo/chosen_quality"] = chosen_quality.mean().item()  
            metrics[f"{prefix}grpo/rejected_quality"] = rejected_quality.mean().item()  
            metrics[f"{prefix}grpo/quality_diff"] = (chosen_quality - rejected_quality).mean().item()  
            
            # 添加各个组件的指标（如果需要更详细的监控）  
            if hasattr(self.reasoning_evaluator, "last_component_scores"):  
                for component, scores in self.reasoning_evaluator.last_component_scores.items():  
                    metrics[f"{prefix}grpo/{component}"] = scores.mean().item()  
    
        return losses.mean(), metrics

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        loss = super().compute_loss(model, inputs, return_outputs)
        if is_transformers_version_equal_to_4_46() and kwargs.pop("num_items_in_batch", False):
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def log(self, logs: Dict[str, float]) -> None:
        r"""
        Log `logs` on the various objects watching training, including stored metrics.
        """
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs)

    @override
    def training_step(self, model, inputs, *args, **kwargs):
        # TODO: sequence_parallel modes other than 'zigzag-ring' may not need dummy forward
        # if os.environ.get("TRACK_TRAINING_DATA", "0") == "1":  
        try:  
            print("执行DPO这里try") 
            # Create log directory  
            log_dir = os.path.join(self.args.output_dir, "training_data_logs")  
            os.makedirs(log_dir, exist_ok=True)  
            
            # Extract batch information  
            batch_size = len(inputs["input_ids"]) // 2  # DPO uses pairs of examples  
            
            # Decode input tokens to get the conversations  
            if hasattr(self, "tokenizer"):  
                chosen_texts = self.tokenizer.batch_decode(  
                    inputs["input_ids"][:batch_size],   
                    skip_special_tokens=True  
                )  
                rejected_texts = self.tokenizer.batch_decode(  
                    inputs["input_ids"][batch_size:],   
                    skip_special_tokens=True  
                )  
                
                # Log to file  
                with open(os.path.join(log_dir, f"step_{self.state.global_step}_data.json"), "w", encoding="utf-8") as f:  
                    json.dump({  
                        "step": self.state.global_step,  
                        "chosen_conversations": chosen_texts,  
                        "rejected_conversations": rejected_texts  
                    }, f, ensure_ascii=False, indent=2)  
        except Exception as e: 
            print("执行DPO这里Exception") 
            print(f"Error logging training data: {e}")  

        if not self._has_dummy_forwarded and model.sequence_parallel_group is not None:
            model.eval()
            with torch.no_grad():
                _ = model(**inputs)
            model.train()
            self._has_dummy_forwarded = True
        return super().training_step(model, inputs, *args, **kwargs)

    @override
    def _get_train_sampler(self):
        if self.model.sequence_parallel_group is not None:
            return SequentialSampler(self.train_dataset)
        else:
            return super()._get_train_sampler()
