# Unofficial Implementation of Animate Anyone

## Overview
This repository contains an simple and unofficial implementation of [Animate Anyone](https://humanaigc.github.io/animate-anyone/). This project is built upon [magic-animate](https://github.com/magic-research/magic-animate/tree/main) and [AnimateDiff](https://github.com/guoyww/AnimateDiff).

## News 🤗🤗🤗
The first training phase basic test passed, currently in training and testing the second phase.

Training may be slow due to GPU shortage.😢

## Note !!!
This project is under continuous development in part-time, there may be bugs in the code, welcome to correct them, I will optimize the code after the pre-trained model is released!

In the current version, we recommend training on 8 or 16 A100,H100 (80G) at 512 or 768 resolution. **Low resolution (256,384) does not give good results!!!(VAE is very poor at reconstruction at low resolution.)**

## Requirements
Same as [magic-animate](https://github.com/magic-research/magic-animate/tree/main).

or you can:
```bash
bash fast_env.sh
```

## ToDo
- [x] **Release Training Code.**
- [ ] DeepSpeed + Accelerator Training
- [ ] **Release Inference Code and unofficial pre-trained weights.**

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

## Acknowledgements
Special thanks to the original authors of the [Animate Anyone](https://humanaigc.github.io/animate-anyone/) project and the contributors to the [magic-animate](https://github.com/magic-research/magic-animate/tree/main) and [AnimateDiff](https://github.com/guoyww/AnimateDiff) repository for their open research and foundational work that inspired this unofficial implementation.

## Email
guoqin@stu.pku.edu.cn
