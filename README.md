# Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios

## [Project Page]() | [Paper]()

This is the official implmentation of the paper xxx.

## Introduction

![workflow](README.assets/workflow.png)

In this work, we propose to emphasize discriminative features for dataset distillation in the complex scenario, i.e. images in complex scenarios are characterized by significant variations in object sizes and the presence of a large amount of class-irrelevant information.

EDF achieves this from supervision and data perspectives via a *Common Pattern Dropout* and a *Discriminative Area Enhancement* module, respectively:

- **Common Pattern Dropout**: We drop low-loss suerpvision as it contains mainly common patterns and weakens the discriminative feature representation.
- **Discriminative Area Enhancement**: We use Grad-CAM activation maps to create pixel gradients which serve as guidance of distillation. As a result, high-activation areas receive more updates.

## Peformance



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

For the detail of Comp-DD, please refer to Section 3 our paper.

We provide fundemental ways to load data, perform distillation, and evaluate.

### Load Data

We provide an interface to load any subset of Comp-DD in `comp-dd/load_data`.  A sample usage is provided below:

```python
data_path = "/path/to/imagenet" 
category = "bird" # must be one of ['bird', 'dog', 'car', 'fish', 'insect', 'snake', 'round', 'music']
subset = "easy" # must be easy or hard
im_size = (128, 128) # by default, we use resolution 128x128
batch_size = 256

channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = load_comp_dd(data_path, category, subset, im_size, batch_size)

image, label = dst_train[0]
```

Explanation of return values:

- `channel`: The number of channles of an image. By default is 3, referring to R, G, B.
- `im_size`: The size of an image. By default is 128x128.
- `num_classes`: The number of classes of the loaded subset. In Comp-DD, we use 10-class subsets.
- `dst_train`: The train dataset. Each element of the dataset is a tuple of an image and its label.
- `dst_test`: The test dataset.
- `class_map`: The mapping from original ImageNet-1K class indices to `[0, num_classes-1]` for each class in the subset.
- `class_map_inv`: The reverse class index mapping.

### Evaluation

After you obatin distilled datasets with your method, you can use our evaluation script to evaluate the performance by running the follow:

```bash
python3 eval.py 
--data_path /path/to/imagenet --data_dir /path/to/syn_images --label_dir /path/to/syn_labels --lr_dir /path/to/syn_lr 
--category bird --subset "easy"
--model ConvNetD5 --lr_net 1e-2
```

We implement the evaluation with differentiable Siamese augmentation. You can disable the augmentation by setting `--dsa False`. 

We do not use knowledge distillation strategy to evaluate the synthetic data. Specifically, the student model is trained by minimizing the Cross-Entropy loss between output logits and labels. 

Moreover, we implement the soft Cross-Entropy loss to cope with soft labels as follows:

```python
def SoftCrossEntropy(inputs, target):
    input_log_likelihood = -torch.nn.functional.log_softmax(inputs, dim=1)
    target_log_likelihood = torch.nn.functional.softmax(target, dim=1)
    batch = inputs.shape[0]
    loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
    return loss
```

## Acknowledgement

Our code is built on [PAD](https://github.com/NUS-HPC-AI-Lab/PAD)
