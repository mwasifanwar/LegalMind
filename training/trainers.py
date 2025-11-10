import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm

class LegalBERTTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('training.learning_rate', 2e-5),
            weight_decay=config.get('training.weight_decay', 0.01)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy
    
    def train(self, epochs=10):
        best_val_accuracy = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.logger.info(f'Epoch {epoch+1}/{epochs}:')
            self.logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            self.logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc
                }, 'models/best_legal_bert.pth')
        
        return best_val_accuracy

class ClauseClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        self.logger = logging.getLogger(__name__)
    
    def train(self, epochs=20):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            self.logger.info(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

class ComplianceTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.risk_criterion = nn.CrossEntropyLoss()
        self.compliance_criterion = nn.BCEWithLogitsLoss()
        self.logger = logging.getLogger(__name__)
    
    def train(self, epochs=15):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                risk_labels = batch['risk_labels'].to(self.device)
                compliance_labels = batch['compliance_labels'].to(self.device)
                
                self.optimizer.zero_grad()
                risk_logits, compliance_logits = self.model(input_ids, attention_mask)
                
                risk_loss = self.risk_criterion(risk_logits, risk_labels)
                compliance_loss = self.compliance_criterion(compliance_logits, compliance_labels)
                loss = risk_loss + compliance_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            self.logger.info(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')