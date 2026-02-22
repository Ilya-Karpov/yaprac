import torch

import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=2, bidirectional=False, 
                 dropout=0.5, pad_idx=0):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, 
                            bidirectional=self.bidirectional, dropout=0.5) # или True bidirect? 
        self.dropout = nn.Dropout(0.5)
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, vocab_size)

    def forward(self, x, hidden=None, lengths=None):
        emb = self.embedding(x)
        if lengths is not None:
            # Режим обучения - используем pack_padded_sequence
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.cpu() if lengths.is_cuda else lengths
            
            # Убеждаемся, что все длины > 0
            original_seq_len = x.size(1)
            lengths = torch.clamp(lengths, min=1, max=original_seq_len)
            
            packed = pack_padded_sequence(input=emb, lengths=lengths, batch_first=True, enforce_sorted=False) #lengths.cpu()
            packed_output, (hidden, cell) = self.lstm(packed)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            if output.size(1) < original_seq_len:
                pad_size = original_seq_len - output.size(1)
                output = torch.nn.functional.pad(output, (0, 0, 0, pad_size), mode='constant', value=0)
        else:
            # режим generate
            output, (hidden, cell) = self.lstm(emb, hidden)
            output = output[:, -1, :]

        output = self.dropout(output)
        out = self.fc(output)

        return out, (hidden, cell)
    
    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0, eos_token_id=None):
        self.eval()
        
        # Подготовка входа
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        
        generated = prompt_ids.clone()
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward на текущей последовательности
                logits, hidden = self.forward(generated, hidden)
                
                # Temperature
                logits = logits / temperature
                
                # Выбор следующего токена
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Добавляем к последовательности
                generated = torch.cat([generated, next_token], dim=1)
                
                # Проверка на eos
                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break
        
        return generated
    