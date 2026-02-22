import torch

from src.eval import compute_rouge_metrics

def train_model(model, optimizer, train_loader, val_loader, tokenizer, device, 
                num_epochs):
    
    train_losses = []
    val_rouge_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            lengths = attention_mask.sum(dim=1)

            optimizer.zero_grad()
            outputs, _ = model(input_ids, hidden=None, lengths=lengths)

            loss = next_token_loss(outputs, input_ids, target_ids, ignore_index=0)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        results, generated_summaries, reference_texts = compute_rouge_metrics(
            model, val_loader, tokenizer, device
        )
        
        val_rouge_scores.append(results)
        
        print(f'Epoch: {epoch} | Train loss: {epoch_loss} | Train rouge1: {results['rouge1']} | Train rouge2: {results['rouge2']}')
        
        # Выводим несколько примеров
        print("\n  Примеры генерации:")
        for i in range(min(1, len(generated_summaries))):
            print(f"    Промпт: {tokenizer.decode(val_loader.dataset[i]['input_ids'][:int(len(val_loader.dataset[i]['input_ids'])*0.75)])}")
            print(f"    Сгенерировано: {generated_summaries[i]}")
            print(f"    Правильно: {reference_texts[i]}")
            print()

    return train_losses, val_rouge_scores

# def next_token_loss(logits, targets, ignore_index=0):
#     logits = logits[:, :-1, :].contiguous()  # убираем последний токен
#     targets = targets[:, 1:].contiguous()    # сдвигаем цели
    
#     loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
#     loss = loss_fn(
#         logits.view(-1, logits.size(-1)),
#         targets.view(-1)
#     )
#     return loss

def next_token_loss(logits, input_ids, target_ids, ignore_index=0):
    batch_size, seq_len, vocab_size = logits.shape

    # non_pad_targets = (target_ids != 0).sum().item()
    # non_pad_logits = (logits.sum(dim=-1) != 0).sum().item()

    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = target_ids.reshape(-1)

    assert logits_flat.size(0) == targets_flat.size(0), \
           f"Shape mismatch: {logits_flat.size(0)} vs {targets_flat.size(0)}"
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fn(logits_flat, targets_flat)
    
    return loss