#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

#seed_lst=(8 13 42 50 60)
#seed_lst=(60 50 42 13 8)
seed_lst=(8)
#task_name_lst=(SST-2 Yelp AGNews TREC MRPC SNLI)  # SST-2 Yelp AGNews TREC MRPC SNLI
task_name_lst=(SNLI)

#sigma_init_lst=(1 0.5)
#learning_rate_lst=(0.5 0.1 0.02)
sigma_init_lst=(6)
learning_rate_lst=(0.2)

alg=CMA
cuda=0


for task_name in "${task_name_lst[@]}"; do
     for learning_rate in "${learning_rate_lst[@]}"; do
          for sigma_init in "${sigma_init_lst[@]}"; do
     		for seed in "${seed_lst[@]}"; do
         		python -u bbt.py --seed $seed --task_name $task_name --alg $alg --sigma_init $sigma_init --learning_rate $learning_rate
     		done
          done
     done
     #python -u test.py --task_name $task_name --cuda $cuda
done


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 
