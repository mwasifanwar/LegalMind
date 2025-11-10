import re
import string
from typing import List, Dict, Any

class TextProcessor:
    def __init__(self):
        self.legal_stop_words = {
            'hereinafter', 'whereas', 'hereby', 'herein', 'hereof',
            'hereto', 'hereunder', 'forthwith', 'pursuant', 'notwithstanding'
        }
        
        self.contraction_map = {
            "won't": "will not",
            "can't": "cannot", 
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'t": " not",
            "'ve": " have",
            "'m": " am"
        }
    
    def clean_text(self, text: str) -> str:
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        text = re.sub(r'\s+', ' ', text)
        
        text = self._expand_contractions(text)
        
        text = text.strip()
        
        return text
    
    def _expand_contractions(self, text: str) -> str:
        for contraction, expansion in self.contraction_map.items():
            text = text.replace(contraction, expansion)
        return text
    
    def tokenize_legal_text(self, text: str) -> List[str]:
        text = self.clean_text(text)
        
        tokens = re.findall(r'\b[\w\']+\b', text)
        
        tokens = [token.lower() for token in tokens if token.lower() not in self.legal_stop_words]
        
        return tokens
    
    def extract_defined_terms(self, text: str) -> Dict[str, str]:
        defined_terms = {}
        
        definition_patterns = [
            r'("([^"]+)"\s*means\s*([^\.]+))',
            r'(term\s+"([^"]+)"\s*shall\s*mean\s*([^\.]+))',
            r'("([^"]+)"\s*shall\s*mean\s*([^\.]+))',
            r'(the\s+term\s+"([^"]+)"\s*refers\s*to\s*([^\.]+))'
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(2)
                definition = match.group(3).strip()
                defined_terms[term] = definition
        
        return defined_terms
    
    def calculate_readability_score(self, text: str) -> float:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = self.tokenize_legal_text(text)
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        complex_words = [word for word in words if self._count_syllables(word) >= 3]
        percent_complex = len(complex_words) / len(words) if words else 0
        
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * percent_complex)
        
        return max(0, min(100, readability_score))
    
    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        
        if word[0] in vowels:
            count += 1
        
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        
        if word.endswith('e'):
            count -= 1
        
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        
        if count == 0:
            count = 1
        
        return count
    
    def extract_dates(self, text: str) -> List[Dict[str, Any]]:
        date_patterns = [
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return dates
    
    def extract_monetary_amounts(self, text: str) -> List[Dict[str, Any]]:
        money_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(USD|dollars)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(euro|EUR|pounds|GBP)'
        ]
        
        amounts = []
        for pattern in money_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amounts.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return amounts
    
    def create_text_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks