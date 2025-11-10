import torch
import torch.nn as nn
from transformers import AutoModel

class ComplianceChecker(nn.Module):
    def __init__(self, model_name="legal-bert"):
        super(ComplianceChecker, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.risk_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.compliance_classifier = nn.Linear(self.bert.config.hidden_size, 5)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        risk_logits = self.risk_classifier(pooled_output)
        compliance_logits = self.compliance_classifier(pooled_output)
        
        return risk_logits, compliance_logits

class RegulationSpecificChecker(nn.Module):
    def __init__(self, num_regulations=8, model_name="legal-bert"):
        super(RegulationSpecificChecker, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        
        self.regulation_heads = nn.ModuleDict({
            'gdpr': nn.Linear(self.bert.config.hidden_size, 3),
            'ccpa': nn.Linear(self.bert.config.hidden_size, 3),
            'sox': nn.Linear(self.bert.config.hidden_size, 3),
            'hippa': nn.Linear(self.bert.config.hidden_size, 3)
        })
        
    def forward(self, input_ids, attention_mask=None, regulation=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        if regulation and regulation in self.regulation_heads:
            logits = self.regulation_heads[regulation](pooled_output)
        else:
            logits = self.regulation_heads['gdpr'](pooled_output)
        
        return logits