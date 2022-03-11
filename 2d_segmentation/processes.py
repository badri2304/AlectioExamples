import os
import random
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from model import UNet
from utils import CityscapeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def getdatasetstate(args={}):
    return {k: k for k in range(2500)}

def train(args, labeled, resume_from, ckpt_file):
    root_dir = './data'
    train_dir = os.path.join(root_dir,'train')
    train_fnms = os.listdir(train_dir)
    
    dataset = CityscapeDataset(train_dir)
    dataset = Subset(dataset, labeled)
    dataloader = DataLoader(dataset, batch_size=args["batch_size"])
    
    model = UNet(3, args["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    predictions, targets = [], []

    for epoch in range(args["epochs"]):
        epoch_loss = 0
        for X,Y in tqdm(dataloader, total=len(dataloader), leave=False):
            X,Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            output = model(X)
            pred = torch.argmax(output, dim=1)
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if epoch ==args["epochs"] - 1:
                predictions.extend(pred.cpu().numpy())
                targets.extend(Y.cpu().numpy())
    
    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file))

    return {"predictions": predictions, "labels": targets}


def test(args, ckpt_file):
    root_dir = './data'
    test_dir = os.path.join(root_dir,'val')
    test_fnms = os.listdir(test_dir)

    dataset = CityscapeDataset(test_dir)
    dataloader = DataLoader(dataset, batch_size=args["batch_size"])

    model = UNet(3, args["num_classes"]).to(device)
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    model.load_state_dict(ckpt["model"])
    model.eval()

    predictions, targets = [], []

    with torch.no_grad():
        for X,Y in tqdm(dataloader, desc="testing"):
            X,Y = X.to(device), Y.to(device)
            output = model(X)
            pred = torch.argmax(output, dim=1)

            predictions.extend(pred.cpu().numpy())
            targets.extend(Y.cpu().numpy())

    return {"predictions": predictions, "labels": targets}

def infer(args, unlabeled, ckpt_file):
    root_dir = './data'
    train_dir = os.path.join(root_dir,'train')
    train_fnms = os.listdir(train_dir)

    dataset = CityscapeDataset(train_dir)
    dataset = Subset(dataset, unlabeled)
    dataloader = DataLoader(dataset, batch_size=1)
    
    model = UNet(3, args["num_classes"]).to(device)
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    model.load_state_dict(ckpt["model"])
    model.eval()

    outputs_fin, k = {}, 0

    with torch.no_grad():
        for X,Y in tqdm(dataloader, desc="Inferring"):
            X,Y = X.to(device), Y.to(device)
            output = model(X)
            pred = torch.argmax(output, dim=1)

            outputs_fin[k] = {}
            outputs_fin[k]["pre_softmax"] = output.cpu().numpy()
            outputs_fin[k]["predicted"] = pred
            k += 1

    return {"outputs" : outputs_fin}


if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)