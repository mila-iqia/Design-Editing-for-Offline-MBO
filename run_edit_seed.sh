
seeds="0 1 2 3 4 5 6 7"

for seed in $seeds; do
  echo $seed
  python design_baselines/diff/edit_new.py --config configs/score_diffusion.cfg --seed $seed --use_gpu --mode 'eval' \
    --task superconductor \
    --save_prefix edit \
    --edit True \
    --t 0.4
done