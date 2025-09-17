#!/bin/bash
# Example: `setsid nohup ./train_test.sh BiRefNet 0,1,2,3,4,5,6,7 0 &>nohup.log &`

experiment_name=${1:-"BSL"}
devices=${2:-"0,1,2,3,4,5,6,7"}

bash train.sh ${experiment_name} ${devices}

devices_test=${3:-0}
bash test.sh ${devices_test}

hostname
