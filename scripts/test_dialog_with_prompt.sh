#!/bin/bash
MODEL_TYPE="cpt-base"
CKPT_PATH="./output/dialog/checkpoint-2700"
SAVE_TO=$CKPT_PATH
TEST_FILE="./data/dialog/6topic/test_test.json"

python test_dialog_with_prompt.py \
  --model_type  ${MODEL_TYPE} \
  --ckpt_path   ${CKPT_PATH} \
  --save_to     ${SAVE_TO} \
  --test_data   ${TEST_FILE}
