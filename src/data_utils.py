import re
import emoji
import torch

import numpy as np
import pandas as pd

#from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


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

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }
    
# добавить lengths???
def collate_fn(batch):
    # список текстов и классов из батча
    texts = [item['text'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    input_ids_stack = torch.stack(texts)
    attention_mask_stack = torch.stack(attention_mask)

    input_ids_input = input_ids_stack[:, :-1]
    input_ids_target = input_ids_stack[:, 1:]

    attention_mask_input = attention_mask_stack[:, :-1]

    # возвращаем преобразованный батч
    return {
        'input_ids': input_ids_input,
        'target_ids': input_ids_target,
        'attention_mask': attention_mask_input
    }