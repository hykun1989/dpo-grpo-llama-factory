# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler
from transformers import Seq2SeqTrainer
from transformers.trainer import _is_peft_model
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        self._has_dummy_forwarded = False

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

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
    def training_step(self, model, inputs, *args, **kwargs):

        # Track training data if enabled  
        # if os.environ.get("TRACK_TRAINING_DATA", "0") == "1":  
        try:  
            print("执行这里try")
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
            print("执行这里Exception") 
            print(f"Error logging training data: {e}")  


        # TODO: sequence_parallel modes other than 'zigzag-ring' may not need dummy forward
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

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        if model.sequence_parallel_group is None:  # no sequence parallel, compute as it is
            loss = super().compute_loss(model, inputs, return_outputs, **kwargs)
        else:
            # compute loss without shift labels, as we have already shifted labels in data processing when using sequence parallel
            _, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="sum")
            logits, labels = outputs["logits"] if isinstance(outputs, dict) else outputs[1], inputs["labels"]
            # Get vocab_size
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                vocab_size = unwrapped_model.base_model.model.config.vocab_size
            else:
                vocab_size = unwrapped_model.config.vocab_size
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)

            # weighted reduce within sequence_parallel_group
            sp_group = model.sequence_parallel_group
            loss = dist.nn.all_reduce(loss, op=dist.ReduceOp.SUM, group=sp_group)
            label_num = (labels != loss_fct.ignore_index).sum()
            label_num = dist.nn.all_reduce(label_num, op=dist.ReduceOp.SUM, group=sp_group)
            loss /= label_num

        # now is single-sequence loss
        # print('loss', loss.shape, loss)

        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
            # other model should not scale the loss
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))
