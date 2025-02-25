#export CUDA_VISIBLE_DEVICES=2
#python gen_model_answer.py --model-path HuggingFaceH4/mistral-7b-sft-beta --model-id mistral-7b-sft-beta
#python gen_judgment.py --mode pairwise-all --model-list mistral-7b-sft-beta beta_ultra_dpo_full_beta0.01_new beta_ultra_rdpo_full_eta0.005_beta0.01_no_decay_new
python gen_judgment.py --mode pairwise-all --model-list mistral-7b-sft-beta beta_ultra_dpo_full_beta0.01_new beta_ultra_rdpo_full_eta0.005_beta0.01_no_decay_new