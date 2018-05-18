#!/bin/bash

declare -a backbones=("resnet50" "resnet101")
declare -a gradient_clip_norms=(5 10)
declare -a learning_rates=(0.01 0.001 0.0001)
declare -a weight_decays=(0.0001)

for i in "${backbones[@]}"
do
    for j in "${gradient_clip_norms[@]}"
    do
        for k in "${learning_rates[@]}"
        do
            for l in "${weight_decays[@]}"
            do
                echo "Training coco.py with hyperparameters:"
                export BACKBONE=$i
                echo "BACKBONE: " "$BACKBONE"
                export GRADIENT_CLIP_NORM=$j
                echo "GRADIENT_CLIP_NORM: " "$GRADIENT_CLIP_NORM"
                export LEARNING_RATE=$k
                echo "LEARNING_RATE: " "$LEARNING_RATE"
                export WEIGHT_DECAY=$l
                echo "WEIGHT_DECAY: " "$WEIGHT_DECAY"
                wandb run samples/coco/coco.py train --model=imagenet --dataset=../coco/
                wait
            done
        done
    done
done
