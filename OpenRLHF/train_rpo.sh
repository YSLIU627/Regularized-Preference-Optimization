# eta = 0.2 for RPO
ETA=0.2
deepspeed --master_port 13678 --module openrlhf.cli.train_dpo \
   --max_len 1024 \
   --dataset argilla/dpo-mix-7k \
   --chosen_key chosen \
   --rejected_key rejected \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain HuggingFaceH4/zephyr-7b-gemma-sft-v0.1\
   --save_path "models/zephyr-7b-gemma-rpo" \
   --beta 0.5 \
   --save_steps 600 \
   --nll_loss_coef ${ETA} \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 3 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-7 \
   --gradient_checkpointing \
   --apply_chat_template \
   --ckpt_path ckpt/ 
