#!/bin/bash
# Run script

# Settings of training & test for different tasks.
experiment_name="$1"

# Train
devices=$2
nproc_per_node=$(echo ${devices%%,} | grep -o "," | wc -l)

to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)

if [ ${to_be_distributed} == "True" ]
then
    # Adapt the nproc_per_node by the number of GPUs. Give 8989 as the default value of master_port.
    echo "Multi-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    torchrun --standalone --nproc_per_node $((nproc_per_node+1)) \
    train.py --experiment_name ${experiment_name} \
        --dist ${to_be_distributed}

else
    echo "Single-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    python train.py --experiment_name ${experiment_name} \
        --dist ${to_be_distributed} \
        --use_accelerate
fi

echo Training finished at $(date)
