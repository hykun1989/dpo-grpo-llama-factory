export DS_SKIP_CUDA_CHECK=1 
export DISABLE_VERSION_CHECK=1  # if necessary
export TRACK_DATA_IDS=1
# dpo
deepspeed --hostfile=hostfile.2nodes src/train.py \
    --stage dpo \
    --do_train \
    --model_name_or_path /lustre/huangyk/model_down/DeepSeek-R1-Distill-Qwen-32B \
    --dataset dpo_zh_stage2_499 \
    --template qwen \
    --finetuning_type lora \
    --output_dir output/debug-0513 \
    --cache_dir .cache \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 24000 \
    --drop_exceed_length_data True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --save_strategy epoch \
    --learning_rate 5e-7 \
    --num_train_epochs 10 \
    --plot_loss \
    --pref_beta 0.3 \
    --lora_rank 16 \
    --do_eval false \
    --pref_loss grpo \
    --use_reasoning_quality true \
    --reasoning_weight 0.3 \
    --format_weight 0.3 \
    --cosine_weight 0.3 \
    --cosine_min_len_value_wrong -0.5 \
    --cosine_max_len_value_wrong 0.0 \
    --cosine_min_len_value_correct 1.0 \
    --cosine_max_len_value_correct 0.5 \
    --cosine_max_len 24000 \
    --save_only_model True \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
    --bf16 True \
    --flash_attn fa2 \
    --gradient_checkpointing True \
    --seed 42 \
    --sequence_parallel_size 8 \
    --packing True \
    --preprocessing_num_workers 32 \
    --report_to tensorboard