export CUDA_VISIBLE_DEVICES=0
#python gen_model_answer.py --model-path HuggingFaceH4/zephyr-7b-beta --model-id zephyr-7b-beta
#python gen_judgment.py --model-list beta_ultra_dpo_full_beta0.01_new beta_ultra_rdpo_full_eta0.01_beta0.01_no_decay_new beta_ultra_rdpo_full_eta0.001_beta0.01_no_decay_new beta_ultra_rdpo_full_eta0.01_beta0.01_stepwise_end0.1_new beta_ultra_rdpo_full_eta0.01_beta0.01_stepwise_end0.5_new beta_ultra_rdpo_full_eta0.008_beta0.01_no_decay_new beta_ultra_rdpo_full_eta0.005_beta0.01_stepwise_end0.5_new beta_ultra_rdpo_full_eta0.005_beta0.01_no_decay_new beta_ultra_rdpo_full_eta0.002_beta0.01_no_decay_new

#python gen_model_answer_gemma.py --model-path HuggingFaceH4/zephyr-7b-beta --model-id zephyr-7b-beta
#python gen_judgment.py --model-list mistral-7b-sft-beta
#python gen_model_answer.py --model-path ../../../model/gemma_dpo --model-id gemma_dpo
#python gen_model_answer.py --model-path ../../../model/gemma_rdpo_eta0.005_no_decay --model-id gemma_rdpo_eta0.005_no_decay
python gen_model_answer.py --model-path HuggingFaceH4/zephyr-7b-gemma-v0.1 --model-id zephyr-7b-gemma-v0.1
python gen_model_answer.py --model-path HuggingFaceH4/zephyr-7b-gemma-sft-v0.1 --model-id zephyr-7b-gemma-sft-v0.1
python gen_judgment.py --model-list zephyr-7b-gemma-v0.1 zephyr-7b-gemma-sft-v0.1 --parallel 10 
