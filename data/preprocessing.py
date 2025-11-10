import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.fitted = False
    
    def preprocess_contract_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'[^\w\s\.\,\;\:\-\$\(\)]', '', text)
        
        text = text.strip()
        
        return text
    
    def encode_clause_labels(self, labels: List[str]) -> np.ndarray:
        if not self.fitted:
            self.label_encoder.fit(labels)
            self.fitted = True
        
        return self.label_encoder.transform(labels)
    
    def decode_clause_labels(self, encoded_labels: np.ndarray) -> List[str]:
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def create_training_pairs(self, contracts_data: List[Dict[str, Any]]) -> tuple:
        texts = []
        labels = []
        
        for contract in contracts_data:
            texts.append(contract['text'])
            
            clause_vector = [0] * len(self.label_encoder.classes_)
            for clause in contract.get('clause_labels', []):
                if clause in self.label_encoder.classes_:
                    idx = list(self.label_encoder.classes_).index(clause)
                    clause_vector[idx] = 1
            
            labels.append(clause_vector)
        
        return texts, labels
    
    def augment_legal_data(self, text: str) -> List[str]:
        augmentations = []
        
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            shuffled_sentences = sentences.copy()
            np.random.shuffle(shuffled_sentences)
            augmentations.append(' '.join(shuffled_sentences))
        
        words = text.split()
        if len(words) > 10:
            delete_positions = np.random.choice(len(words), size=min(5, len(words)//10), replace=False)
            augmented_words = [word for i, word in enumerate(words) if i not in delete_positions]
            augmentations.append(' '.join(augmented_words))
        
        return augmentations
    
    def normalize_risk_scores(self, risk_scores: List[float]) -> List[float]:
        if not risk_scores:
            return []
        
        min_score = min(risk_scores)
        max_score = max(risk_scores)
        
        if max_score == min_score:
            return [0.5] * len(risk_scores)
        
        normalized = [(score - min_score) / (max_score - min_score) for score in risk_scores]
        return normalized
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        features = {}
        
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        legal_terms = ['hereinafter', 'whereas', 'hereby', 'notwithstanding']
        features['legal_term_count'] = sum(1 for term in legal_terms if term in text.lower())
        
        defined_terms = re.findall(r'"([^"]+)"', text)
        features['defined_terms_count'] = len(defined_terms)
        
        monetary_pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        features['monetary_mentions'] = len(re.findall(monetary_pattern, text))
        
        date_pattern = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'
        features['date_mentions'] = len(re.findall(date_pattern, text))
        
        return features