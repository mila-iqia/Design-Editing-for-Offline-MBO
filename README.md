# Design-Editing-with-SDE-for-Offline-MBO

## Install dependencies
**Install MujoCo and mujoco-py**
Follow the instructions from [here](https://github.com/openai/mujoco-py).
If you have never installed `Mujoco` before, please also make sure you have `.mujoco` directory under your home directory and include `mujoco200` inside.

**Create a conda environment and install the dependencies**
```bash
conda create -n sde python=3.7 --file requirements.txt
conda activate sde
```

## Directory structure
```
configs/
├── score_diffusion.cfg  # Use score matching. Other configs are not used.
design_baselines/
├── diff/
│   ├── lib/  # Contains the implementation diffusion models
│   ├── edit.py  # Contains the implementation of the design editing algorithm, need to trian diffusion model on both source and target distribution first
│   ├── trainer.py  # Contains the implmentation of training a diffusion model and evaluate the vanilla DDOM models. Includes training for the pseudo-target distribution with top k candidates in the offline dataset
│   ├── nets.py  # Contains the implementation of the neural networks used in the diffusion models
│   ├── util.py  # Contains the utility functions for the diffusion models
│   ├── grad.py  # Contains the implementation of the gradient ascent process to generate pseudo-target distribution (train proxies and generate designs)
│   ├── my_model.py  # Contains the implementation of the neural networks used for training proxies
│   ├── utils.py  # Contains the utility functions for gradient ascent process to generate pseudo-target distribution
│   ├── ##  Other files are not used
experiements/  # Contains the training logs and models checkpoints
npy/  # Contains the maximum score and minimum score of the offline dataset, used for calculating the normalized score
results/  # Contains the results of the evaluation experiments
requirements.txt  # Contains the dependencies
run_diff.sh  # Script to run training and evaluation for different seeds and different tasks
setup.py  # Contains the dependencies
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

**Run training experiment on source distribution**
```bash
python design_baselines/diff/trainer.py --config {a config file from configs/} --seed {seed} --use_gpu --mode 'train' --task {task name}
```
For example, to run the experiment for `superconductor` with `score-matching`:
```bash
python design_baselines/diff/trainer.py --config configs/score_diffusion.cfg --seed 0 --use_gpu --mode 'train' --task superconductor
```

**Run training experiment on pseudo-target distribution with top k candidates from the offline dataset**
```bash
python design_baselines/diff/trainer.py --config {a config file from configs/} --seed {seed} --use_gpu --mode 'train' --task {task name} --size {size of how many designs to use from the offline dataset}
```
For example, to run the experiment for `superconductor` with `score-matching`:
```bash
python design_baselines/diff/trainer.py --config configs/score_diffusion.cfg --seed 0 --use_gpu --mode 'train' --task superconductor --size 512
```

** Run training experiment for proxy models**
The task names are not exactly the same as training the diffusion models. We directly use the exact task names here:
- Superconfuctor-RandomForest-v0
- AntMorphology-Exact-v0
- HopperController-Exact-v0
- DKittyMorphology-Exact-v0
- TFBind8-Exact-v0
- TFBind10-Exact-v0
- CIFARNAS-Exact-v0

```bash
python design_baselines/diff/grad.py --seed {seed} --task {task name} --mode 'train' --store_path {path to store the trained models}
```
For example, to run the experiment for `superconductor`:
```bash
python design_baselines/diff/grad.py --seed 0 --task Superconfuctor-RandomForest-v0 --mode 'train' --store_path experiments/superconductor
```

** Run training experiment for generating designs**
```bash
python design_baselines/diff/grad.py --seed {seed} --task {task name} --mode 'generate' --Tmax {how many steps for gradient ascent, 100 for discrete tasks amd 200 for continuous tasks} --ft_lr {learning rate for gradient ascent, 1e-1 for discrete tasks and 1e-3 for continuous tasks} --store_path {path to load the trained models and save the generated designs}
```
For example, to run the experiment for `superconductor`:
```bash
python design_baselines/diff/grad.py --seed 0 --task Superconfuctor-RandomForest-v0 --mode 'generate' --Tmax 100 --ft_lr 1e-1 --store_path experiments/superconductor
```
For example, to run the experiment for `ant`:
```bash
python design_baselines/diff/grad.py --seed 0 --task AntMorphology-Exact-v0 --mode 'generate' --Tmax 200 --ft_lr 1e-3 --store_path experiments/ant
```

## Need to add how to train DDOM on the proxy's generated designs

## Evaluation
**Run evaluation experiment for a single DDOM model**
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

**Run evaluation experiment for the edited designs after the editing process**
Note that the source and target DDOM models need to be trained first.
When running the editing process, the source DDOM is automatically selected as the last run DDOM model, and the target DDOM should be specified.
```bash
python design_baselines/diff/edit.py --config {a config file from configs/} --seed {seed} --use_gpu --mode 'eval' --task {task name} --suffix "max_ds_conditioning"  --target_checkpoint_path {path to the trained DDOM model of the target_checkpoint_path}
```
For example, to run the experiment for `superconductor` with `score-matching`:
```bash
python design_baselines/diff/edit.py --config configs/score_diffusion.cfg --seed 0 --use_gpu --mode 'eval' --task superconductor --suffix "max_ds_conditioning" --target_checkpoint_path experiments/superconductor/checkpoint_512.pth
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
