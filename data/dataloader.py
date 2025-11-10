import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any

class LegalDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        
        if os.path.exists(os.path.join(self.data_dir, 'contracts.json')):
            with open(os.path.join(self.data_dir, 'contracts.json'), 'r') as f:
                contracts_data = json.load(f)
            
            for contract in contracts_data:
                samples.append({
                    'text': contract.get('text', ''),
                    'clause_labels': contract.get('clause_labels', []),
                    'risk_level': contract.get('risk_level', 'low'),
                    'metadata': contract.get('metadata', {})
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class LegalDataLoader:
    def __init__(self, config):
        self.config = config
    
    def create_data_loader(self, data_dir, batch_size=8, shuffle=True, num_workers=0):
        dataset = LegalDataset(data_dir)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    def load_contract_data(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def save_analysis_results(self, results, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def export_to_csv(self, data, filepath):
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def load_regulation_rules(self, regulation_name):
        regulation_files = {
            'gdpr': './data/regulations/gdpr_rules.json',
            'ccpa': './data/regulations/ccpa_rules.json',
            'sox': './data/regulations/sox_rules.json',
            'hippa': './data/regulations/hippa_rules.json'
        }
        
        filepath = regulation_files.get(regulation_name.lower())
        if filepath and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        
        return {}

class ContractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }