#!/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="0 1 2 3 4 5 6 7"

for seed in $seeds; do
  echo $seed
  echo $TASK
  python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK
  python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --suffix "max_ds_conditioning" > "results/${TASK}_seed_${seed}_eval_output.txt"
done
