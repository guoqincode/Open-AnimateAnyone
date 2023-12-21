# Unofficial Implementation of Animate Anyone

## Overview
This repository contains an simple and unofficial implementation of [Animate Anyone](https://humanaigc.github.io/animate-anyone/). This project is built upon [magic-animate](https://github.com/magic-research/magic-animate/tree/main) and [AnimateDiff](https://github.com/guoyww/AnimateDiff).

If you find this repository helpful, please consider giving us a star⭐! 

## News 🤗🤗🤗
The first training phase basic test passed, currently in training and testing the second phase.

~~Training may be slow due to GPU shortage.😢~~

It only takes a few days to release the weights.😄

## Note !!!
This project is under continuous development in part-time, there may be bugs in the code, welcome to correct them, I will optimize the code after the pre-trained model is released!

In the current version, we recommend training on 8 or 16 A100,H100 (80G) at 512 or 768 resolution. **Low resolution (256,384) does not give good results!!!(VAE is very poor at reconstruction at low resolution.)**

## ToDo
- [x] **Release Training Code.**
- [x] **Release Inference Code.** 
- [ ] **Release Unofficial Pre-trained Weights. <font color="red">(Note:Train on public datasets instead of large-scale private datasets, just for academic research.🤗)</font>**
- [ ] **Release Gradio Demo.**
- [ ] DeepSpeed + Accelerator Training.


## Requirements
Same as [magic-animate](https://github.com/magic-research/magic-animate/tree/main).

or you can:
```bash
bash fast_env.sh
```

## Training

#### First Stage

```python
torchrun --nnodes=2 --nproc_per_node=8 train.py --config configs/training/train_stage_1.yaml
```

#### Second Stage

```python
torchrun --nnodes=2 --nproc_per_node=8 train.py --config configs/training/train_stage_2.yaml
```

## Inference

#### First Stage

```python
python3 -m pipelines.animation_stage_1 --config configs/prompts/animation_stage_1.yaml
```

#### Second Stage

```python
python3 -m pipelines.animation_stage_2 --config configs/prompts/animation_stage_2.yaml
```

## Acknowledgements
Special thanks to the original authors of the [Animate Anyone](https://humanaigc.github.io/animate-anyone/) project and the contributors to the [magic-animate](https://github.com/magic-research/magic-animate/tree/main) and [AnimateDiff](https://github.com/guoyww/AnimateDiff) repository for their open research and foundational work that inspired this unofficial implementation.

## Email
guoqin@stu.pku.edu.cn

My response may be slow, please don't ask me nonsense questions.
