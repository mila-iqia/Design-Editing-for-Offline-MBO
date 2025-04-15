

# Design Editing for Offline Model-based Optimization 
This repository contains the official implementation for the paper **"Design Editing for Offline Model-based Optimization"**, which is accepted by [**Transactions on Machine Learning Research (TMLR) 2025**](https://openreview.net/forum?id=OPFnpl7KiF) and [**ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy**]((https://openreview.net/forum?id=fF1KXgAhKN)). See below for more details.

  
## Install dependencies  
**Install MujoCo and mujoco-py**  
Follow the instructions from [here](https://github.com/openai/mujoco-py).  
If you have never installed `Mujoco` before, please also make sure you have `.mujoco` directory under your home directory and include `mujoco200` inside.  
  
**Create a conda environment and install the dependencies**  
```bash  
conda create --name DEMO --file requirements.txt
conda activate DEMO
```  
  
## Directory structure  
```  
configs/  
├── score_diffusion.cfg  # Use score matching. Other configs are not used.
design_baselines/  
├── diff/  
│   ├── lib/  # Contains the implementation diffusion models.
│   ├── edit_new.py  # Contains the implementation of the design editing algorithm, need to train diffusion model first.
│   ├── trainer.py  # Contains the implmentation of training a diffusion model
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
setup.py  # Contains the dependencies  
```  
  
## Training  
**Available tasks:**  
- `superconductor`: Superconductor-RandomForest-v0  
- `ant`: AntMorphology-Exact-v0  
- `dkitty`: DKittyMorphology-Exact-v0  
- `levy`: Levy-Exact-v0
- `tf-bind-8`: TFBind8-Exact-v0  
- `tf-bind-10`: TFBind10-Exact-v0  
- `nas`: CIFARNAS-Exact-v0  
 
 **Run training experiment for proxy models**
```bash  
python design_baselines/diff/grad.py --seed {seed} --mode 'train' \
	--task {task name} \
	--store_path {path to store the trained models}
```  
For example, to run the experiment for `superconductor` using random seed 123:  
```bash  
python design_baselines/diff/grad.py --seed 123 --mode 'train' \
	--task Superconfuctor-RandomForest-v0 \
	--store_path experiments/superconductor
```
or directly:
```bash
bash run_grad_train.sh
```

**Run training experiment for optimizing offline design samples** 
```bash  
python design_baselines/diff/grad.py --seed {seed} --mode 'design' \
	--task {task name} \
	--store_path {path to store the trained models}
```  
For example, to run the experiment for `superconductor` using random seed 123:  
```bash  
python design_baselines/diff/grad.py --seed 123 --mode 'design' \
	--task Superconfuctor-RandomForest-v0 \
	--store_path experiments/superconductor
```
or directly:
```bash
bash run_grad_gen.sh
```

**Run training experiment on source distribution**  
```bash
python design_baselines/diff/trainer.py --config {a config file from configs/} --seed {seed} --use_gpu --mode 'train' \
	--task {task name} \
	--is_target False
```
For example, to run the experiment for `superconductor` with `score-matching` using random seed 123:  
```bash  
python design_baselines/diff/trainer.py --config configs/score_diffusion.cfg --seed 123 --use_gpu --mode 'train' \
	--task superconductor \
	--is_target False
```
or directly:
```bash  
bash run_train_source.sh
```
  
## Design Edit  
**Run evaluation on the gradient ascent samples**  
```bash  
python design_baselines/diff/edit_new.py --config {a config file from configs/} --seed {seed} --use_gpu --mode 'eval' \
  --task {task name} \  
  --save_prefix {prefix of saved json file} \  
  --edit False
```  
For example, to run the experiment for `superconductor` with `score-matching` using random seed 123 and save as `src_123.json`:  
```bash  
python design_baselines/diff/edit_new.py --config configs/score_diffusion.cfg --seed 123 --use_gpu --mode 'eval' \  
  --task superconductor \  
  --save_prefix src \  
  --edit False
```
  
**Run evaluate on the edited samples**  
Note that the diffusion models need to be trained first, and the corresponding checkpoint paths of the diffusion models need to be specified in `edit_new.py`.
```bash  
python design_baselines/diff/edit_new.py --config {a config file from configs/} --seed {seed} --use_gpu --mode 'eval' \
  --task {task name} \  
  --save_prefix {prefix of saved json file} \  
  --edit True
```  
For example, to run the experiment for `superconductor` with `score-matching` using random seed 123 and save as `edit_123.json`:  
```bash  
python design_baselines/diff/edit_new.py --config configs/score_diffusion.cfg --seed 123 --use_gpu --mode 'eval' \  
  --task superconductor \  
  --save_prefix edit \  
  --edit True
```
or directly:
```bash
bash run_edit.sh
```
  
**Run evaluation on different seeds**
```bash  
bash run_edit_seed.sh
```

## Citation
Please cite our paper if you find it is useful in your research:
```bibtex
@inproceedings{yuan2025design,
	title={Design Editing for Offline Model-based Optimization},
	author={Ye Yuan and Youyuan Zhang and Can Chen and Haolun Wu and Melody Zixuan Li and Jianmo Li and James J. Clark and Xue Liu},
	booktitle={ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy},
	year={2025},
	url={https://openreview.net/forum?id=fF1KXgAhKN}
}
```
