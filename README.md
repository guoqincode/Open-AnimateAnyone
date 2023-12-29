# Unofficial Implementation of Animate Anyone

If you find this repository helpful, please consider giving us a star⭐!

## Overview
This repository contains an simple and unofficial implementation of [Animate Anyone](https://humanaigc.github.io/animate-anyone/). This project is built upon [magic-animate](https://github.com/magic-research/magic-animate/tree/main) and [AnimateDiff](https://github.com/guoyww/AnimateDiff). This implementation is first developed by [Qin Guo](https://github.com/guoqincode) and then assisted by [Zhenzhi Wang](https://zhenzhiwang.github.io/).

## News 🤗🤗🤗
The first training phase basic test passed, currently in training and testing the second phase.

~~Training may be slow due to GPU shortage.😢~~

It only takes a few days to release the weights.😄

## Sample of Result on UBC-fashion dataset
### Stage 1
The current version of the face still has some artifacts.  This model is trained on the UBC dataset rather than a large-scale dataset.
<table class="center">
    <tr><td><img src="./assets/stage1/1.png"></td><td><img src="./assets/stage1/2.png"></td></tr>
    <tr><td><img src="./assets/stage1/3.png"></td><td><img src="./assets/stage1/8.png"></td></tr>
    <tr><td><img src="./assets/stage1/9.png"></td><td><img src="./assets/stage1/10.png"></td></tr>
    <tr><td><img src="./assets/stage1/4.png"></td><td><img src="./assets/stage1/5.png"></td></tr>
    <tr><td><img src="./assets/stage1/6.png"></td><td><img src="./assets/stage1/7.png"></td></tr>

</table>
<p style="margin-left: 2em; margin-top: -1em"></p>

### Stage 2
The training of stage2 is challenging due to artifacts in the background. We select one of our best results here, and are still working on it. An important point is to ensure that training and inference resolution is consistent.
<table class="center">
    <tr><td><img src="./assets/stage2/1.gif"></td></tr>

</table>
<p style="margin-left: 2em; margin-top: -1em"></p>


## Note !!!
This project is under continuous development in part-time, there may be bugs in the code, welcome to correct them, I will optimize the code after the pre-trained model is released!

In the current version, we recommend training on 8 or 16 A100,H100 (80G) at 512 or 768 resolution. **Low resolution (256,384) does not give good results!!!(VAE is very poor at reconstruction at low resolution.)**

## ToDo
- [x] **Release Training Code.**
- [x] **Release Inference Code.** 
- [ ] **Release Unofficial Pre-trained Weights. <font color="red">(Note:Train on public datasets instead of large-scale private datasets, just for academic research.🤗)</font>**
- [ ] **Release Gradio Demo.**

## Requirements

```bash
bash fast_env.sh
```

## 🎬Gradio Demo (will publish with weights.)
```python
python3 -m demo.gradio_animate
```

If you only have a GPU with 24 GB of VRAM, I recommend inference at resolution 512 and below.


## Training
### Original AnimateAnyone Architecture (It is difficult to control pose when training on a small dataset.)
#### First Stage

```python
torchrun --nnodes=8 --nproc_per_node=8 train.py --config configs/training/train_stage_1.yaml
```

#### Second Stage

```python
torchrun --nnodes=8 --nproc_per_node=8 train.py --config configs/training/train_stage_2.yaml
```

### Our Method (A more dense pose control scheme, the number of parameters is still small.)
```python
torchrun --nnodes=8 --nproc_per_node=8 train_hack.py --config configs/training/train_stage_1.yaml
```

#### Second Stage

```python
torchrun --nnodes=8 --nproc_per_node=8 train_hack.py --config configs/training/train_stage_2.yaml
```


## Acknowledgements
Special thanks to the original authors of the [Animate Anyone](https://humanaigc.github.io/animate-anyone/) project and the contributors to the [magic-animate](https://github.com/magic-research/magic-animate/tree/main) and [AnimateDiff](https://github.com/guoyww/AnimateDiff) repository for their open research and foundational work that inspired this unofficial implementation.
