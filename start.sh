#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DATA_DIR='/share/nas169/jethrowang/CADA-GAN/hat_asr_aligned/channel_encoder'
BATCH_SIZE=2
NUM_WORKERS=16
MAX_EPOCHS=30
ENCODER_NAME='conformer_cat'
LOSS_NAME='amsoftmax'
LEARNING_RATE=0.001
WARMUP_STEP=327999
WEIGHT_DECAY=0.0000001
STEP_SIZE=4
GAMMA=0.5
SAVE_DIR='./results'
INPUT_LAYER=conv2d2
CHECKPOINT_PATH='./channel_encoder'

python main.py \
  --data_dir $DATA_DIR \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --max_epochs $MAX_EPOCHS \
  --encoder_name $ENCODER_NAME \
  --loss_name $LOSS_NAME \
  --learning_rate $LEARNING_RATE \
  --warmup_step $WARMUP_STEP \
  --weight_decay $WEIGHT_DECAY \
  --step_size $STEP_SIZE \
  --gamma $GAMMA \
  --save_dir $SAVE_DIR \
  --input_layer $INPUT_LAYER \
  --checkpoint_path $CHECKPOINT_PATH