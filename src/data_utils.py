import re
import emoji
import torch

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

special_chars = (
    ("à", "a"),
    ("á", "a"),
    ("â", "a"),
    ("ã", "a"),
    ("ä", "a"),
    ("å", "a"),
    ("ç", "c"),
    ("è", "e"),
    ("é", "e"),
    ("ê", "e"),
    ("ë", "e"),
    ("ì", "i"),
    ("í", "i"),
    ("î", "i"),
    ("ï", "i"),
    ("ñ", "n"),
    ("ò", "o"),
    ("ó", "o"),
    ("ô", "o"),
    ("õ", "o"),
    ("ö", "o"),
    ("ø", "o"),
    ("ù", "u"),
    ("ú", "u"),
    ("û", "u"),
    ("ü", "u"),
    ("ý", "y"),
    ("ÿ", "y"),
    (':)', ' happy '),
    (':(', ' sad '),
    (";)", " wink "),
    (":D", " laugh "),
    (":P", " tongue "),
    (":'(", " cry "),
    ("<3", " heart "),
)

def nonstandart(text):
    for special_char, char in special_chars :
        text = text.replace(special_char, char)
    return text

def clean_string(text):
    # приведение к нижнему регистру
    text = text.lower()
    # замена нестандартных символов
    text = nonstandart(text)
    # удаление ссылок
    text = re.sub(r'https?://\S+|www\.\S+|\S+\.(ru|com|org|net)\S*', '', text)
    # удаление упоминаний
    text = re.sub(r'@\w+', '', text)
    # удаление хэштегов
    text = re.sub(r'#(\w+)', r'\1', text)
    # удаление эмодзи
    text = emoji.replace_emoji(text, replace='')
    # удаление всего, кроме латинских букв, цифр и пробелов
    text = re.sub(r'[^a-zа-яё0-9\s]', '', text)
    # удаление дублирующихся пробелов, удаление пробелов по краям
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def to_tokenizer(dataframe):
    tokenized_lengths = []
    for text in dataframe['clean_text'].head(10000):
        tokens = tokenizer.tokenize(text)
        tokenized_lengths.append(len(tokens))

    MAX_LENGTH = int(np.percentile(tokenized_lengths, 95))

    encoded = tokenizer(
        dataframe['clean_text'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

    return encoded
