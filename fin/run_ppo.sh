#!/bin/bash

python train_ppo.py \
  --max_train_steps 10000 \
  --evaluate_freq 500 \
  --save_freq 1 \
  --batch_size 2048 \
  --mini_batch_size 128 \
  --lr_a 3e-4 \
  --lr_c 3e-4 \
  --gamma 0.99 \
  --lamda 0.95 \
  --epsilon 0.2 \
  --K_epochs 5 \
  --entropy_coef 0.01 \
  --use_adv_norm True \
  --use_state_norm True \
  --use_reward_norm False \
  --use_reward_scaling True \
  --use_lr_decay False \
  --use_grad_clip True \
  --use_orthogonal_init True \
  --set_adam_eps 1e-5 \
  --use_tanh True
