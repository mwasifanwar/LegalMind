import torch
import numpy as np
import logging
import os
import json
from datetime import datetime

def setup_logging(log_dir='./logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'legalmind_{datetime.now().strftime("%Y%m%d")}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    return epoch, metrics

def calculate_metrics(predictions, targets):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def create_directories():
    dirs = ['./models', './data', './logs', './results', './exports']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def format_legal_text(text, max_line_length=80):
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        if len(' '.join(current_line + [word])) <= max_line_length:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)

def save_analysis_results(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_tensors(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def load_analysis_results(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def validate_contract_structure(analysis_results):
    required_clauses = ['parties', 'term', 'payment']
    found_clauses = [c['clause_type'] for c in analysis_results.get('clause_analysis', [])]
    
    missing_clauses = [clause for clause in required_clauses if clause not in found_clauses]
    
    return {
        'is_valid': len(missing_clauses) == 0,
        'missing_clauses': missing_clauses,
        'found_clauses': found_clauses
    }