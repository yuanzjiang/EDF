# Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios

## [Project Page]() | [Paper]()

This is the official implmentation of the paper xxx.

![workflow](README.assets/workflow.png)

In this work, we propose to emphasize discriminative features for dataset distillation in the complex scenario, i.e. images in complex scenarios are characterized by significant variations in object sizes and the presence of a large amount of class-irrelevant information.

EDF achieves this from supervision and data perspectives via a *Common Pattern Dropout* and a *Discriminative Area Enhancement* module, respectively:

- **Common Pattern Dropout**: We drop low-loss suerpvision as it contains mainly common patterns and weakens the discriminative feature representation.
- **Discriminative Area Enhancement**: We use Grad-CAM activation maps to create pixel gradients which serve as guidance of distillation. As a result, high-activation areas receive more updates.

## Getting Started

Create environment as follows:

```bash
conda env create -f environment.yaml
conda activate dd
```

## Train Expert Trajectories

To train expert trajectories, you can run

```bash
bash scripts/buffer.sh
```

In the script, we demo with the "ImageNette" subset.  Change the argument `--subset` to other subsets when training expert trajectories on them.

For the list of available subsets, please refer to `utils/utils_gsam`.

## Distill

To perform distillation, please run:

```bash
bash scripts/distill_in1k_ipc1.sh # for ipc1
bash scripts/distill_in1k_ipc1.sh # for ipc10
bash scripts/distill_in1k_ipc1.sh # for ipc50
```

Similarly, the sample scripts provided use "ImageNette" for demo. You can change the subset easily as follows:

```bash
cd distill
CFG="../configs/ImageNet/SUBSET/ConvIN/IPC1.yaml" # replace the SUBSET with the one you want to distill
python3 edf_distill.py --cfg $CFG
```

Hyper-parameters in each config file in `configs` are the ones used in EDF main experiments. Feel free to play around with other hyper-parameters for distillation by modifying the corresponding config file.

## Evaluation

By default, we perform evaluation along after every 500/1000 iterations. If you want to evaluate distilled explicitly, you can run

```bash
cd distill
python3 evaluation.py --lr_dir=path_to_lr --data_dir=path_to_images --label_dir=path_to_labels
```

In our paper, we also use knowledge distillation to ensure a fair comparison against methods that integrate knowledge distillation during evaluation. For detailed implementation, please refer to the official codebase of [SRe2L](https://github.com/VILA-Lab/SRe2L.git) and [RDED](https://github.com/LINs-lab/RDED.git).

## Comp-DD Benchmark

We are excited to release the Complex Dataset Distillation benchmark (Comp-DD), an early effort for the community to explore effective dataset distillation methods in the complex scenario.

For the detail of Comp-DD, please refer to our paper Section 3.

We provide fundemental ways to load data, perform distillation, and evaluate.

### Load Data

We provide an interface to load any subset of Comp-DD.

