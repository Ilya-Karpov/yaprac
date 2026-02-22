import evaluate
import torch

from tqdm import tqdm
from transformers import pipeline

def compute_rouge_metrics(model, dataloader, tokenizer, device, max_new_tokens=20):
    model.eval()
    generated_summaries = []  # нагенерированное
    reference_texts = []  # (target)
    
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        #attention_mask = batch['attention_mask'].to(device)

        #lengths = attention_mask.sum(dim=1)
        
        for i in range(input_ids.size(0)):
            # удаление паддинга в каждом примере
            example_ids = input_ids[i]
            non_pad = (example_ids != 0).sum()
            example_ids = example_ids[:non_pad]
            
            if len(example_ids) < 4:
                continue
            
            # Разделяем на промпт (первые 3/4) и таргет (последние 1/4)
            split_point = int(len(example_ids) * 0.75)
            prompt_ids = example_ids[:split_point]
            target_ids = example_ids[split_point:]
            
            if len(target_ids) == 0:
                continue
            
            # Подготавливаем промпт для генерации
            prompt_ids_batch = prompt_ids.unsqueeze(0).to(device)  # (1, prompt_len)
            
            with torch.no_grad():
                summary_ids = model.generate(
                    prompt_ids=prompt_ids_batch,
                    max_new_tokens=min(max_new_tokens, len(target_ids))
                )  # сгенерируйте саммари
            
            # Декодирование
            generated_part = summary_ids[0, split_point:split_point + max_new_tokens]
            summary = tokenizer.decode(generated_part, skip_special_tokens=True)
            
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)

            generated_summaries.append(summary)
            reference_texts.append(target_text)

    rouge = evaluate.load("rouge")
    # Подсчёт метрик
    results = rouge.compute(predictions=generated_summaries, references=reference_texts)
    
    return results, generated_summaries, reference_texts


def compute_rouge_trans(model, tokenizer, dataloader, device, 
                              max_new_tokens=15, top_p=0.95):

    model.eval()
    generated_texts = []
    reference_texts = []
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, pad_token_id=tokenizer.pad_token_id)
    
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']  # BERT токены
        
        for i in range(input_ids.size(0)):
            example_ids = input_ids[i]
            non_pad = (example_ids != 0).sum()
            example_ids = example_ids[:non_pad]
            
            if len(example_ids) < 4:
                continue
            
            full_text = tokenizer.decode(example_ids, skip_special_tokens=True)
            
            words = full_text.split()
            if len(words) < 4:
                continue
                
            split_point = int(len(words) * 0.75)
            prompt_words = words[:split_point]
            target_words = words[split_point:]
            
            prompt_text = ' '.join(prompt_words)
            target_text = ' '.join(target_words)
            
            out = generator(prompt_text, max_new_tokens=min(max_new_tokens, len(target_words)), 
                            do_sample=True, top_p=top_p, num_return_sequences=1)
            
            generated_full = out[0]['generated_text']
            
            generated_part = generated_full[len(prompt_text):].strip()
            
            generated_texts.append(generated_part)
            reference_texts.append(target_text)
    
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=generated_texts, references=reference_texts)
    
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    for i in range(min(5, len(generated_texts))):
        print(f"  начало: {prompt_text}")
        print(f"  создано: {generated_texts[i]}")
        print(f"  таргет: {reference_texts[i]}")
    
    return results, generated_texts, reference_texts