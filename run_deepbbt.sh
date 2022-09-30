#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

seed_lst=(8 13 42 50 60) # 8 13 42 50 60
task_name_lst=(MRPC)  # SST-2 Yelp AGNews TREC MRPC SNLI
device=cuda:1
cuda=1
model_name=roberta-large
model_path=transformer_model/roberta-large
n_prompt_tokens=50
intrinsic_dim=500
k_shot=16
loss_type=ce  # hinge ce
random_proj=normal
sigma1=1
sigma2=0.2
popsize=20
bound=0
budget=8000
print_every=50
eval_every=100
data_dir=datasets_augment_pos_small_final  # datasets_augment_pos_final  # datasets_augment_pos_final
# --use_rlprompt --multiVerbalizer

for task_name in "${task_name_lst[@]}"; do
    for seed in "${seed_lst[@]}"; do
        python -u deepbbt.py --instruction --data_dir $data_dir --seed $seed --task_name $task_name --device $device --budget $budget --model_name $model_name --model_path $model_path
    done
    # python -u test.py --task_name $task_name --cuda $cuda
done


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 