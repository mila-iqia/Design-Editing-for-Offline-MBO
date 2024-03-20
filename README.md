# Design-Editing-with-SDE-for-Offline-MBO

## Install dependencies
**Install MujoCo and mujoco-py**
Follow the instructions from [here](https://github.com/openai/mujoco-py).
If you have never installed `Mujoco` before, please also make sure you have `.mujoco` directory under your home directory and include `mujoco200` inside.

**Create a conda environment and install the dependencies**
```bash
conda create -n sde python=3.7
conda activate sde
pip install -r requirements.txt
```

## Training
**Available tasks:**
- `superconductor`: Superconductor-RandomForest-v0
- `ant`: AntMorphology-Exact-v0
- `hopper`: HopperController-Exact-v0
- `dkitty`: DKittyMorphology-Exact-v0
- `tf-bind-8`: TFBind8-Exact-v0
- `tf-bind-10`: TFBind10-Exact-v0
- `nas`: CIFARNAS-Exact-v0

**Run training experiment**
```bash
python design_baselines/diff/trainer.py --config {a config file from configs/} --seed {seed} --use_gpu --mode 'train' --task {task name}
```
For example, to run the experiment for `superconductor` with `score-matching`:
```bash
python design_baselines/diff/trainer.py --config configs/score_diffusion.cfg --seed 0 --use_gpu --mode 'train' --task superconductor
```

## Evaluation
**Run evaluation experiment**
```bash
python design_baselines/diff/trainer.py --config {a config file from configs/} --seed {seed} --use_gpu --mode 'eval' --task {task name}  --suffix "max_ds_conditioning"
```
For example, to run the experiment for `superconductor` with `score-matching`:
```bash
python design_baselines/diff/trainer.py --config configs/score_diffusion.cfg --seed 0 --use_gpu --mode 'eval' --task superconductor --suffix "max_ds_conditioning"
```
It displays the maximum and median score of the `512` generated designs, like:
```
Max Score:  0.9839299
Median Score:  0.41181505
```

## Run training and evaluation for different seeds
**Run training and evaluation experiment for different seeds**
```bash
./run_diff_eval.sh {a config file from configs/} {task name}
```
For example, to run the experiment for `superconductor` with `score-matching`:
```bash
./run_diff_eval.sh configs/score_diffusion.cfg superconductor
```
