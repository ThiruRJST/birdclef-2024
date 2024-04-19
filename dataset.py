from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
from hypers import HyperParams
from albumentations.pytorch import ToTensorV2
from albumentations import Compose


class CLEFMelspec_v1(Dataset):
    def __init__(self, root_path, paths, labels, mode="train") -> None:
        self.root = root_path
        self.mode = mode
        self.paths = paths
        self.labels = labels
        self.transforms = Compose([ToTensorV2()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        p = f"{self.root}/{self.paths[index].replace('.ogg', '.png')}"
        image = cv2.imread(p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        image = self.transforms(image=image)["image"]

        return image, label


if __name__ == "__main__":
    labels = [x.strip() for x in open("labels.txt", "r").readlines()]
    data = pd.read_csv("train_metadata.csv")
    data["class_index"] = data.primary_label.apply(lambda x: labels.index(x))

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        data.filename.values,
        data.class_index.values,
        test_size=HyperParams.test_size,
        stratify=data.class_index.values,
    )

    train_dataset = CLEFMelspec_v1(HyperParams.root_path, train_paths, train_labels)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    imgs, labels = next(iter(train_loader))
    print(imgs.shape)
