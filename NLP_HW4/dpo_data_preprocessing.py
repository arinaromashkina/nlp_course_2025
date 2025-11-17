import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import random
import re


class DPODataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.examples = []
        
        skipped = 0
        for item in data:
            chosen_text = item['chosen']
            rejected_text = item['rejected']
            
            prompt = self.extract_prompt(chosen_text, rejected_text)
            
            if prompt:
                self.examples.append({
                    'prompt': prompt,
                    'chosen': chosen_text,
                    'rejected': rejected_text,
                })
            else:
                skipped += 1

    
    def extract_prompt(self, chosen: str, rejected: str) -> Optional[str]:
        pattern = r'\n\nAssistant:'
        
        chosen_matches = list(re.finditer(pattern, chosen))
        rejected_matches = list(re.finditer(pattern, rejected))
        
        if not chosen_matches or not rejected_matches:
            return None
        
        last_match = chosen_matches[-1]
        split_pos = last_match.start()
        
        chosen_prompt = chosen[:split_pos].strip()
        rejected_prompt = rejected[:rejected_matches[-1].start()].strip()
        
        if chosen_prompt != rejected_prompt:
            if len(chosen_prompt) < len(rejected_prompt):
                return chosen_prompt
            else:
                return rejected_prompt
        
        return chosen_prompt
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return self.tokenize_pair(
            example['prompt'],
            example['chosen'],
            example['rejected'],
        )
    
    def tokenize_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict[str, torch.Tensor]:
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=True,
        )
        
        prompt_length = len(prompt_tokens['input_ids'])

        chosen_tokens = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        
        rejected_tokens = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        vocab_size = self.tokenizer.vocab_size
        
        chosen_ids = chosen_tokens['input_ids']
        rejected_ids = rejected_tokens['input_ids']

        chosen_ids = [min(t, vocab_size - 1) for t in chosen_ids]
        rejected_ids = [min(t, vocab_size - 1) for t in rejected_ids]

        chosen_labels = chosen_ids.copy()
        chosen_labels[:prompt_length] = [-100] * min(prompt_length, len(chosen_labels))
        
        rejected_labels = rejected_ids.copy()
        rejected_labels[:prompt_length] = [-100] * min(prompt_length, len(rejected_labels))
        
        return {
            'chosen_input_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'chosen_attention_mask': torch.tensor(chosen_tokens['attention_mask'], dtype=torch.long),
            'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
            'rejected_input_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'rejected_attention_mask': torch.tensor(rejected_tokens['attention_mask'], dtype=torch.long),
            'rejected_labels': torch.tensor(rejected_labels, dtype=torch.long),
            'prompt': prompt,
        }


def collate_fn_dpo(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    max_chosen_len = max(len(x['chosen_input_ids']) for x in batch)
    max_rejected_len = max(len(x['rejected_input_ids']) for x in batch)
    max_len = max(max_chosen_len, max_rejected_len)
    
    chosen_input_ids = []
    chosen_attention_mask = []
    chosen_labels = []
    rejected_input_ids = []
    rejected_attention_mask = []
    rejected_labels = []
    prompts = []
    
    for item in batch:

        chosen_pad_len = max_len - len(item['chosen_input_ids'])
        chosen_input_ids.append(
            torch.cat([
                item['chosen_input_ids'],
                torch.full((chosen_pad_len,), pad_token_id, dtype=torch.long)
            ])
        )
        chosen_attention_mask.append(
            torch.cat([
                item['chosen_attention_mask'],
                torch.zeros(chosen_pad_len, dtype=torch.long)
            ])
        )
        chosen_labels.append(
            torch.cat([
                item['chosen_labels'],
                torch.full((chosen_pad_len,), -100, dtype=torch.long)
            ])
        )
        

        rejected_pad_len = max_len - len(item['rejected_input_ids'])
        rejected_input_ids.append(
            torch.cat([
                item['rejected_input_ids'],
                torch.full((rejected_pad_len,), pad_token_id, dtype=torch.long)
            ])
        )
        rejected_attention_mask.append(
            torch.cat([
                item['rejected_attention_mask'],
                torch.zeros(rejected_pad_len, dtype=torch.long)
            ])
        )
        rejected_labels.append(
            torch.cat([
                item['rejected_labels'],
                torch.full((rejected_pad_len,), -100, dtype=torch.long)
            ])
        )
        
        prompts.append(item['prompt'])

    return {
        'input_ids': torch.cat([
            torch.stack(chosen_input_ids),
            torch.stack(rejected_input_ids)
        ], dim=0),
        'attention_mask': torch.cat([
            torch.stack(chosen_attention_mask),
            torch.stack(rejected_attention_mask)
        ], dim=0),
        'labels': torch.cat([
            torch.stack(chosen_labels),
            torch.stack(rejected_labels)
        ], dim=0),
        'prompts': prompts,
    }


def prepare_tokenizer_and_model_vocab(
    model_name: str,
    model = None,
):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_added = len(tokenizer) - tokenizer.vocab_size
    if num_added > 0:
       
        actual_vocab_size = len(tokenizer)
 
    else:
        actual_vocab_size = tokenizer.vocab_size
    
   
    if model is not None:
        model_vocab_size = model.config.vocab_size
        model_embed_size = model.get_input_embeddings().weight.shape[0]
        
    
        if model_embed_size != actual_vocab_size:
            
            model.resize_token_embeddings(actual_vocab_size)
    
    return tokenizer, actual_vocab_size


def load_dpo_data(
    tokenizer,
    max_length: int = 512,
    max_prompt_length: int = 256,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
) -> Tuple[DPODataset, DPODataset]:
    
    dataset = load_dataset("Anthropic/hh-rlhf")
    
    train_data = dataset['train']
    test_data = dataset['test']
    
    if num_train_samples:
        indices = random.sample(range(len(train_data)), min(num_train_samples, len(train_data)))
        train_data = train_data.select(indices)
    
    if num_val_samples:
        indices = random.sample(range(len(test_data)), min(num_val_samples, len(test_data)))
        test_data = test_data.select(indices)
    
    train_dataset = DPODataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )
    
    val_dataset = DPODataset(
        data=test_data,
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )
    
    return train_dataset, val_dataset


def create_dpo_dataloaders(
    train_dataset: DPODataset,
    val_dataset: DPODataset,
    batch_size: int = 4,
    num_workers: int = 2,
    pad_token_id: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn_dpo(x, pad_token_id),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn_dpo(x, pad_token_id),
        pin_memory=True,
    )
    
    return train_loader, val_loader
