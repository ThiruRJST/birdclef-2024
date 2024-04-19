from hypers import HyperParams
import torch
import timm
import numpy as np
from tqdm import tqdm
import random
import os
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassROC
from torchsummary import summary
from dataset import CLEFMelspec_v1
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def seed_everything(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_resnet():
    model = timm.create_model(model_name=HyperParams.Arch)
    in_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(in_ftrs, HyperParams.num_classes))
    return model


def train(epoch, trainloader, train_loss, optimizer):

    model.train()
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    running_train_loss = 0.0

    for i, (image, label) in pbar:
        image = image.to("cuda:0").float()
        label = label.to("cuda:0").long()

        optimizer.zero_grad()

        outputs = model(image)
        loss = train_loss(outputs, label)
        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()
    
    train_epoch_loss = running_train_loss / len(trainloader)
    return train_epoch_loss


def val(epoch, valloader, val_loss):

    model.eval()
    pbar = tqdm(enumerate(valloader), total=len(valloader))
    running_test_loss = 0.0

    for i, (image, label) in pbar:
        image = image.to("cuda:0").float()
        label = label.to("cuda:0").long()
        
        with torch.no_grad():
            outputs = model(image)
        loss = val_loss(outputs, label)
        running_test_loss += loss.item()
    
    test_epoch_loss = running_test_loss / len(valloader)
    return test_epoch_loss

if __name__ == "__main__":
    print(f"These are the Hyperparameter you have used: {HyperParams()}")
    seed_everything()

    model = load_resnet()
    summary(model, (3, 1954, 256))

    #dataset
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
    test_dataset = CLEFMelspec_v1(HyperParams.root_path, test_paths, test_labels)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    train_loss = nn.CrossEntropyLoss()
    val_loss = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=HyperParams.lr)

    for e in range(HyperParams.epochs):

        print("***** Training *****")
        train_epoch_loss = train(epoch=e, trainloader=train_loader, train_loss=train_loss, optimizer=optimizer)

        print("***** Testing *****")
        test_epoch_loss = val(epoch=e, valloader=test_loader, val_loss=val_loss)

        print(f"Epoch: {e} || Train Loss: {train_epoch_loss} || Test Loss: {test_epoch_loss}")

    

