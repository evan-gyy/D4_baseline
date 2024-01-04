#!/bin/bash
  # -m torch.distributed.launch --nproc_per_node=$nproc --nnodes=1 --node_rank=0 --master_port "${port}" \
# main.py \

per_device_train_batch_size=24
per_device_eval_batch_size=16
gradient_accumulation_steps=4

settings="wo_act"
MODEL_TYPE=cpt-large
MODEL_PATH="/home/hy/models/cpt-large"
train_data="./data/dialog/6topic/train_test.json"
valid_data="./data/dialog/6topic/val_test.json"
test_data="./data/dialog/6topic/test_test.json"
output_dir="./output/${MODEL_TYPE}_${settings}"
result_dir="./result/"

port=49181
quit=0
device=1
nproc=1

while [ "$quit" -ne 1 ]; do
  netstat -a | grep $port >> /dev/null
  if [ $? -gt 0 ]; then
    quit=1
  else
    port=$(expr $port + 1)
  fi
done

CUDA_VISIBLE_DEVICES=$device python main.py \
  --output_dir  ${output_dir} \
  --do_train    true \
  --do_eval     true \
  --do_predict  true \
  --evaluation_strategy     steps \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size  ${per_device_eval_batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --learning_rate               1e-5 \
  --weight_decay                1e-6 \
  --num_train_epochs            30 \
  --lr_scheduler_type           cosine \
  --warmup_steps                100 \
  --logging_steps               40 \
  --save_steps                  500 \
  --eval_steps                  500 \
  --save_total_limit            1 \
  --load_best_model_at_end      true \
  --seed                        42 \
  --dataloader_num_workers      3 \
  --disable_tqdm                false \
  --label_smoothing_factor      0 \
  --ddp_find_unused_parameters  true \
  --dataloader_pin_memory       false \
  \
  --metric_for_best_model       bleu_2 \
  --greater_is_better           true \
  \
  --predict_with_generate       true \
  --generation_max_length       128 \
  --generation_num_beams        4 \
  \
  --model_type    "${MODEL_TYPE}" \
  --model_path    "${MODEL_PATH}" \
  --settings      "${settings}"   \
  --result_dir    "${result_dir}" \
  \
  --max_len       512 \
  --add_portrait  true \
  --task          dialog \
  --train_text    ${train_data} \
  --dev_text      ${valid_data} \
  --test_text     ${test_data}
