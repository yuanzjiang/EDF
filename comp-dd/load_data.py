import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


class Comp_DD_Config:
    birds_hard = [14, 19, 91, 15, 13, 95, 10, 16, 20, 12]

    birds_easy = [130, 140, 83, 142, 134, 139, 88, 90, 144, 21]

    cars_hard = [802, 660, 829, 627, 609, 575, 573, 867, 734, 408]

    cars_easy = [581, 535, 717, 817, 511, 436, 656, 757, 661, 751]

    dogs_hard = [239, 238, 247, 195, 151, 266, 157, 181, 154, 217]

    dogs_easy = [193, 180, 227, 167, 246, 248, 224, 177, 269, 252]

    fish_hard = [0, 391, 389, 395, 4, 396, 2, 394, 50, 103]

    fish_easy = [5, 1, 6, 397, 390, 3, 392, 393, 148, 147]

    snake_hard = [46, 42, 41, 40, 43, 58, 65, 59, 53, 52]

    snake_easy = [64, 57, 55, 56, 61, 62, 68, 67, 54, 60]

    insect_hard = [301, 326, 324, 305, 325, 300, 318, 320, 70, 322]

    insect_easy = [323, 314, 310, 309, 306, 308, 302, 313, 303, 315]

    round_hard = [852, 805, 736, 892, 429, 574, 722, 522, 752, 971]

    round_easy = [417, 560, 953, 826, 426, 981, 430, 890, 746, 768]

    music_hard = [420, 594, 420, 579, 402, 776, 486, 401, 546, 558]

    music_easy = [542, 687, 494, 566, 642, 513, 875, 683, 889, 881]

    dict = {
        "birds_hard": birds_hard,
        'birds_easy': birds_easy,
        'dogs_hard': dogs_hard,
        'dogs_easy': dogs_easy,
        'cars_hard': cars_hard,
        'cars_easy': cars_easy,
        'fish_hard': fish_hard,
        'fish_easy': fish_easy,
        'snake_hard': snake_hard,
        'snake_easy': snake_easy,
        'insect_hard': insect_hard,
        'insect_easy': insect_easy,
        'round_hard': round_hard,
        'round_easy': round_easy,
        'music_hard': music_hard,
        'music_easy': music_easy
    }


config = Comp_DD_Config()


def create_simple_tensor_dataset(dataset, class_map):
    images_all = []
    labels_all = []

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    simple_dataset = TensorDataset(images_all, labels_all)
    return simple_dataset


def load_comp_dd(data_path, category, subset, im_size, batch_size):

    channel = 3
    num_classes = 10
    subset_name = category + '_' + subset

    if subset_name not in config.dict:
        print("Invalid category or subset")
        exit(0)

    config.img_net_classes = config.dict[subset]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std),
                                    transforms.Resize(im_size),
                                    transforms.CenterCrop(im_size)])

    dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
    dst_train_dict = {c: torch.utils.data.Subset(dst_train, np.squeeze(
        np.argwhere(np.equal(dst_train.targets, config.img_net_classes[c])))) for c in
                      range(len(config.img_net_classes))}
    dst_train = torch.utils.data.Subset(dst_train,
                                        np.squeeze(np.argwhere(np.isin(dst_train.targets, config.img_net_classes))))
    loader_train_dict = {
        c: torch.utils.data.DataLoader(dst_train_dict[c], batch_size=batch_size, shuffle=True, num_workers=16) for c in
        range(len(config.img_net_classes))}
    dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
    dst_test = torch.utils.data.Subset(dst_test,
                                       np.squeeze(np.argwhere(np.isin(dst_test.targets, config.img_net_classes))))
    for c in range(len(config.img_net_classes)):
        dst_test.dataset.targets[dst_test.dataset.targets == config.img_net_classes[c]] = c
        dst_train.dataset.targets[dst_train.dataset.targets == config.img_net_classes[c]] = c

    class_map = {x: i for i, x in enumerate(config.img_net_classes)}
    class_map_inv = {i: x for i, x in enumerate(config.img_net_classes)}

    dst_train = create_simple_tensor_dataset(dst_train, class_map)
    dst_test = create_simple_tensor_dataset(dst_test, class_map)
    return channel, num_classes, dst_train, dst_test, class_map, class_map_inv
