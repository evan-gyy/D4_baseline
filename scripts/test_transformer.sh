#!/bin/bash

EPOCH=""
MODEL_PATH="../output/dialog/checkpoint-2700"
OUTPUT_PATH=$MODEL_PATH
TEST_FILE="../data/dialog/6topic/test_test.json"

python transformer/generate.py \
	--model_name $MODEL_PATH \
	--output_path $OUTPUT_PATH \
	--test_file $TEST_FILE