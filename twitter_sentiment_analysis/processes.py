import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset

from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig

from utils import flat_accuracy

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def getdatasetstate(args={}):
    return {k: k for k in range(40000)}

def train(args, labeled, resume_from, ckpt_file):
    # batch_size = args["train_batch_size"]
    # lr = args["lr"]
    # epochs = args["epochs"]

    batch_size, lr, epochs = 32, 2e-5, 2
    
    ids, masks, labels = torch.load("train_ids.pt"), torch.load("train_masks.pt"), torch.load("train_labels.pt")
    
    train_dataset = TensorDataset(ids, masks, labels)
    train_dataset = Subset(train_dataset, labeled)
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    )    
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    predictions, targets = [], []

    if resume_from is not None:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate()

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            output = model(b_input_ids, token_type_ids = None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
            loss = output.loss
            logits = output.logits
            total_train_loss += loss.item()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if epoch == epochs - 1:
                logits = logits.detach().cpu().numpy()
                preds = np.argmax(logits, axis=1)
                predictions.extend(preds)
                targets.extend(b_labels.cpu().numpy())

    print("Finished Training. Saving the model as {}".format(ckpt_file))
    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file))

    return {"predictions": predictions, "labels": targets}

    
def test(args, ckpt_file):
    # batch_size = args["test_batch_size"]
    ids, masks, labels = torch.load("test_ids.pt"), torch.load("test_masks.pt"), torch.load("test_labels.pt")
    test_dataset = TensorDataset(ids, masks, labels)
    batch_size = 32
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    predictions, targets = [], []
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    ) 
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    for batch in tqdm(test_loader, desc="Testing"):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(b_input_ids, token_type_ids = None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
            loss = output.loss
            logits = output.logits

        logits = logits.detach().cpu().numpy()
        
        predictions.extend(np.argmax(logits, axis=1))
        targets.extend(b_labels.cpu().numpy())

    return {"predictions": predictions, "labels": targets}

def infer(args, unlabeled, ckpt_file):
    ids, masks, labels = torch.load("train_ids.pt"), torch.load("train_masks.pt"), torch.load("train_labels.pt")
    train_dataset = TensorDataset(ids, masks, labels)
    train_dataset = Subset(train_dataset, unlabeled)
    unlabeled_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    ) 

    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    outputs_fin, k = {}, 0
    for batch in tqdm(unlabeled_loader, desc="Inferring"):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(b_input_ids, token_type_ids = None, attention_mask=b_input_mask, return_dict=True)
            logits = output.logits
            _, predicted = torch.max(logits, 1)

            print("pre_softmax: ", logits[0].cpu().numpy())


            outputs_fin[k] = {}
            outputs_fin[k]["pre_softmax"] = logits[0].cpu().numpy()
            outputs_fin[k]["predicted"] = predicted
            k += 1

    return {"outputs" : outputs_fin}

if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)