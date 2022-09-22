#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

seed_lst=(60) # 8 13 42 50 60
task_name_lst=(TREC)  # SST-2 Yelp AGNews TREC MRPC SNLI
device=cuda:0
cuda=0
model_name=roberta-large
model_path=roberta-large
n_prompt_tokens=50
intrinsic_dim=500
k_shot=16
loss_type=ce
random_proj=normal # normal trunc_normal xavier_normal kaiming_normal
sigma1=1
sigma2=0.2
popsize=20
bound=0
budget=8000
print_every=50
eval_every=100

for task_name in "${task_name_lst[@]}"; do
    for seed in "${seed_lst[@]}"; do
        python -u deepbbt.py --seed $seed --task_name $task_name --device $device --budget $budget --model_name $model_name --model_path $model_path
    done
    # python -u test.py --task_name $task_name --cuda $cuda
done


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 