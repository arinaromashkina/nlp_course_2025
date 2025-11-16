import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import random
import re


class HHRLHFDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
        prompt_loss_weight: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_loss_weight = prompt_loss_weight
        self.examples = []
        
        for item in data:
            text = item['chosen']

            prompt, response = self.split_prompt_response(text)
            
            if prompt and response:
                self.examples.append({
                    'prompt': prompt,
                    'response': response,
                    'full_text': text,
                })

    
    def split_prompt_response(self, text: str) -> Tuple[str, str]:
        pattern = r'\n\nAssistant:'
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return None, None

        last_match = matches[-1]
        split_pos = last_match.start()
        
        prompt = text[:split_pos].strip()
        response = text[split_pos:].strip()

        if len(prompt) < 10 or len(response) < 10:
            return None, None
        
        return prompt, response
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]

        prompt_encoded = self.tokenizer(
            example['prompt'],
            truncation=True,
            max_length=self.max_length - 100,
            add_special_tokens=True,
        )
        
        response_encoded = self.tokenizer(
            example['response'],
            truncation=True,
            max_length=100,
            add_special_tokens=False,
        )
        
        input_ids = prompt_encoded['input_ids'] + response_encoded['input_ids']
        attention_mask = prompt_encoded['attention_mask'] + response_encoded['attention_mask']
        
        if self.prompt_loss_weight == 0.0:
            labels = [-100] * len(prompt_encoded['input_ids']) + response_encoded['input_ids']
        else:
            labels = input_ids.copy()

        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        labels = labels[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def collate_fn_hh(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(len(x['input_ids']) for x in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        curr_len = len(item['input_ids'])
        padding_length = max_len - curr_len

        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.full((padding_length,), pad_token_id, dtype=torch.long)
        ])

        padded_attention_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(padding_length, dtype=torch.long)
        ])

        padded_labels = torch.cat([
            item['labels'],
            torch.full((padding_length,), -100, dtype=torch.long)
        ])
        
        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        labels.append(padded_labels)
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
    }


def load_hh_rlhf_data(
    tokenizer_name: str = "EleutherAI/pythia-1.4b",
    max_length: int = 512,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
    prompt_loss_weight: float = 0.0,
) -> Tuple[Dataset, Dataset, AutoTokenizer]:

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset("Anthropic/hh-rlhf")

    train_data = dataset['train']
    test_data = dataset['test']
    
    if num_train_samples:
        indices = random.sample(range(len(train_data)), min(num_train_samples, len(train_data)))
        train_data = train_data.select(indices)
    
    if num_val_samples:
        indices = random.sample(range(len(test_data)), min(num_val_samples, len(test_data)))
        test_data = test_data.select(indices)

    train_dataset = HHRLHFDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt_loss_weight=prompt_loss_weight,
    )
    
    val_dataset = HHRLHFDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt_loss_weight=prompt_loss_weight,
    )
    
    return train_dataset, val_dataset, tokenizer


def create_hh_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 2,
    pad_token_id: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn_hh(x, pad_token_id),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn_hh(x, pad_token_id),
        pin_memory=True,
    )
    
    return train_loader, val_loader


def visualize_example(dataset: HHRLHFDataset, idx: int = 0):
    example = dataset.examples[idx]
    
    print("="*80)
    print("EXAMPLE VISUALIZATION")
    print("PROMPT")
    print(example['prompt'])
    print("RESPONSE")
    print(example['response'])
    print("\n" + "="*80)

    tokenized = dataset[idx]
    print("\nTokenized shapes:")
    print(f"input_ids: {tokenized['input_ids'].shape}")
    print(f"attention_mask: {tokenized['attention_mask'].shape}")
    print(f"labels: {tokenized['attention_mask'].shape}")

    num_prompt_tokens = (tokenized['labels'] == -100).sum().item()
    num_response_tokens = (tokenized['labels'] != -100).sum().item()
 
    print(f"Prompt tokens (masked): {num_prompt_tokens}")
    print(f"Response tokens (trained): {num_response_tokens}")
