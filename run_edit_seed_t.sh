
seeds="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49"
ts="0.001 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.999"

for t in $ts;do
  for seed in $seeds; do
    echo $t
    echo $seed
    python design_baselines/diff/edit_new.py --config configs/score_diffusion.cfg --seed $seed --use_gpu --mode 'eval' \
      --task superconductor \
      --save_prefix ablate \
      --edit True \
      --t $t \
      --suffix "max_ds_conditioning"
  done
done