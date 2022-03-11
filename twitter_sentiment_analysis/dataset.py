import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import random_split

from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def get_data(path, tokenizer):
    df = pd.read_csv(path).fillna("none")
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    sentences = df['review'].values
    labels = df['sentiment'].values

    input_ids = []
    attention_masks = []

    for sent in tqdm(sentences, total=len(sentences)):
        encode_dict = tokenizer.encode_plus(
                        sent, 
                        add_special_tokens = True,
                        max_length = 64,
                        padding = 'max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation = True
        )

        input_ids.append(encode_dict['input_ids'])
        attention_masks.append(encode_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


train_path = './data/train.csv'
ids, masks, labels = get_data(train_path, tokenizer)

torch.save(ids, "train_ids.pt")
torch.save(masks, "train_masks.pt")
torch.save(labels, "train_labels.pt")

test_path = './data/test.csv'
ids, masks, labels = get_data(test_path, tokenizer)

torch.save(ids, "test_ids.pt")
torch.save(masks, "test_masks.pt")
torch.save(labels, "test_labels.pt")

