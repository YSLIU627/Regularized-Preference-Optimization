# Regularized-Preference-Optimization

This repository contains the code for our NeurIPS 2024 poster paper: [Provably Mitigating Overoptimization in RLHF: Your SFT Loss is Implicitly an Adversarial Regularizer](https://arxiv.org/abs/2405.16436). 

In this study, we explore two approaches to incorporating the supervised fine-tuning (SFT) loss as a regularizer:

1. **Cumulative SFT Loss**: This method calculates the cumulative SFT loss over all unmasked tokens in the selected response.
2. **Average SFT Loss**: This method computes the average SFT loss across all unmasked tokens in the chosen responses.

We implement these approaches by integrating the [Alignment Handbook Codebase](https://github.com/huggingface/alignment-handbook) for the cumulative loss and the [OpenRLHF Codebase](https://github.com/OpenRLHF/OpenRLHF) for the average loss. 

In the main chapter of our paper (https://arxiv.org/pdf/2405.16436), we use the cumulative loss while in the Appendix F we report the performance of the average SFT loss. It is observed that using the average SFT loss would improve the model reasoning ability while using the cumulative SFT loss can improve the chat ability. This phenomenon may be explained by the fact the chat response is much longer than the reasoning response in the Ultrafeedback dataset. Generally speaking, the average SFT loss is better and it is implemented in OpenRLHF as the default method.
