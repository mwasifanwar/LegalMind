import torch
import torch.nn as nn
from transformers import AutoModel

class LegalBERT(nn.Module):
    def __init__(self, num_labels=9, model_name="legal-bert"):
        super(LegalBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class LegalBERTForSequence(nn.Module):
    def __init__(self, model_name="legal-bert"):
        super(LegalBERTForSequence, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.sequence_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.sequence_classifier(sequence_output)
        return logits.squeeze(-1)