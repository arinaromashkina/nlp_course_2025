import torch
from transformers import AutoTokenizer
from typing import List, Dict
import re


class DialogueGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda',
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        
        self.model.eval()
    
    def generate_response(self, prompt: str, verbose: bool = False) -> str:
        if not prompt.strip().endswith("Assistant:"):
            prompt = prompt.strip() + "\n\nAssistant:"

        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024,
        ).to(self.device)
        
        prompt_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][prompt_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        response = self.clean_response(response)
        
        return response
    
    def clean_response(self, response: str) -> str:
        response = response.strip()
        for marker in ["\n\nHuman:", "\n\nAssistant:"]:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        return response
    
    def chat(self, user_message: str, history: str = "") -> str:
        if history:
            prompt = history + f"\n\nHuman: {user_message}\n\nAssistant:"
        else:
            prompt = f"\n\nHuman: {user_message}\n\nAssistant:"

        response = self.generate_response(prompt)
        
        return response


def test_model_on_examples(
    model,
    tokenizer,
    test_examples: List[Dict[str, str]],
    device: str = 'cuda',
) -> List[Dict]:
    generator = DialogueGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    
    results = []
    
    for i, example in enumerate(test_examples):
        print(f"\n{'='*80}")
        print(f"Example {i+1}/{len(test_examples)}")
        print(f"{'='*80}")
        
        prompt = example['prompt']
        expected = example.get('expected_response', 'N/A')
        
        print("PROMPT")
        print(prompt)

        generated = generator.generate_response(prompt)
        
        print("EXPECTED RESPONSE")
        print(expected)
        
        print("GENERATED RESPONSE")
        print(generated)
        
        results.append({
            'prompt': prompt,
            'expected_response': expected,
            'generated_response': generated,
        })
    
    return results


def create_test_examples_from_dataset(dataset, num_examples: int = 5) -> List[Dict]:
    import random
    
    indices = random.sample(range(len(dataset.examples)), min(num_examples, len(dataset.examples)))
    
    test_examples = []
    for idx in indices:
        example = dataset.examples[idx]
        test_examples.append({
            'prompt': example['prompt'] + "\n\nAssistant:",
            'expected_response': example['response'].replace("\n\nAssistant:", "").strip(),
        })
    
    return test_examples
