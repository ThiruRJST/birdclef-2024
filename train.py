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

def seed_everything(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   

def load_resnet():
    model = timm.create_model(model_name=HyperParams.Arch)
    in_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_ftrs, HyperParams.num_classes)
    )
    return model
if __name__ == "__main__":
    print(f"These are the Hyperparameter you have used: {HyperParams()}")
    seed_everything()
    model = load_resnet()
    summary(model, (3,1954,256))

