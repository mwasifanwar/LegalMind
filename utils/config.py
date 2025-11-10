import yaml
import os

class Config:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self):
        return {
            'model': {
                'legal_bert': {
                    'model_name': 'nlpaueb/legal-bert-base-uncased',
                    'max_length': 512,
                    'batch_size': 16
                },
                'summarization': {
                    'model_name': 'mrm8488/legal-t5-base',
                    'max_input_length': 1024,
                    'max_output_length': 150
                }
            },
            'processing': {
                'document': {
                    'max_file_size': 10485760,
                    'supported_formats': ['.pdf', '.docx', '.txt']
                },
                'text': {
                    'chunk_size': 1000,
                    'overlap': 100,
                    'min_segment_length': 50
                }
            },
            'analysis': {
                'risk': {
                    'high_risk_threshold': 0.7,
                    'critical_risk_threshold': 0.9
                },
                'compliance': {
                    'enabled_regulations': ['gdpr', 'ccpa', 'sox', 'hippa']
                }
            },
            'api': {
                'host': 'localhost',
                'port': 8000,
                'debug': True
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def update(self, key, value):
        keys = key.split('.')
        config_ref = self.config
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        config_ref[keys[-1]] = value
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f)