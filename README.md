# Unofficial Implementation of Animate Anyone

## Overview
This repository contains an simple and unofficial implementation of "Animate Anyone". This project is built upon the concepts and initial work found at [magic-animate](https://github.com/magic-research/magic-animate/tree/main).

## ToDo
- [x] **Release Training Code.**
- [ ] **Release Inference Code.**
- [ ] **Release unofficial pre-trained weights.**
- [ ] **Data Collection and Release (Within Legal Boundaries)**: Efforts to collect and refine the dataset for further training and improvements are ongoing, with the intention to release it publicly, adhering to legal constraints.

## Training

#### First Stage

```python
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1.yaml
```

#### Second Stage

```python
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_2.yaml
```

## Acknowledgements
Special thanks to the original authors of the "Animate Anyone" project and the contributors to the [magic-animate](https://github.com/magic-research/magic-animate/tree/main) repository for their open research and foundational work that inspired this unofficial implementation.
