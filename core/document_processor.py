import re
import PyPDF2
import docx
import torch
from transformers import AutoTokenizer
from typing import List, Dict, Any

class DocumentProcessor:
    def __init__(self, model_name="legal-bert"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 512
        
    def extract_text(self, file_path: str) -> str:
        if file_path.endswith('.pdf'):
            return self._extract_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._extract_from_docx(file_path)
        elif file_path.endswith('.txt'):
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_from_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def segment_document(self, text: str) -> List[Dict[str, Any]]:
        segments = []
        
        sections = self._split_into_sections(text)
        
        for i, section in enumerate(sections):
            if len(section.strip()) > 50:
                segments.append({
                    'id': i,
                    'text': section.strip(),
                    'type': self._classify_section_type(section),
                    'start_char': text.find(section),
                    'end_char': text.find(section) + len(section)
                })
        
        return segments
    
    def _split_into_sections(self, text: str) -> List[str]:
        section_patterns = [
            r'\n\s*(ARTICLE|SECTION)\s+[IVXLCDM0-9]+[\.\)]?\s*\n',
            r'\n\s*[0-9]+\.\s+',
            r'\n\s*\([a-z]\)\s+',
            r'\n\s*[A-Z][A-Z\s]+\n'
        ]
        
        combined_pattern = '|'.join(section_patterns)
        sections = re.split(combined_pattern, text)
        
        return [section for section in sections if section and len(section.strip()) > 0]
    
    def _classify_section_type(self, text: str) -> str:
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['party', 'agreement between', 'between']):
            return 'parties'
        elif any(term in text_lower for term in ['term', 'duration', 'effective date']):
            return 'term'
        elif any(term in text_lower for term in ['payment', 'fee', 'compensation', 'price']):
            return 'payment'
        elif any(term in text_lower for term in ['confidential', 'proprietary', 'nda']):
            return 'confidentiality'
        elif any(term in text_lower for term in ['warranty', 'guarantee', 'representation']):
            return 'warranties'
        elif any(term in text_lower for term in ['liability', 'indemnification', 'hold harmless']):
            return 'liability'
        elif any(term in text_lower for term in ['termination', 'breach', 'default']):
            return 'termination'
        elif any(term in text_lower for term in ['governing law', 'jurisdiction', 'venue']):
            return 'governing_law'
        elif any(term in text_lower for term in ['intellectual property', 'ip', 'copyright']):
            return 'ip'
        else:
            return 'other'
    
    def tokenize_text(self, text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        if max_length is None:
            max_length = self.max_length
        
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoding
    
    def preprocess_contract(self, file_path: str) -> Dict[str, Any]:
        raw_text = self.extract_text(file_path)
        segments = self.segment_document(raw_text)
        
        processed_segments = []
        for segment in segments:
            encoding = self.tokenize_text(segment['text'])
            processed_segments.append({
                **segment,
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            })
        
        return {
            'raw_text': raw_text,
            'segments': processed_segments,
            'total_segments': len(processed_segments),
            'metadata': {
                'file_path': file_path,
                'total_length': len(raw_text),
                'avg_segment_length': sum(len(s['text']) for s in processed_segments) / len(processed_segments)
            }
        }
    
    def extract_clauses(self, text: str) -> List[Dict[str, Any]]:
        clause_patterns = {
            'indemnification': r'indemnif(y|ies|ication).*?\.',
            'limitation_of_liability': r'limitation.*?liability.*?\.',
            'confidentiality': r'confidential.*?\.',
            'termination': r'terminat(e|ion).*?\.',
            'governing_law': r'govern(ing)?.*?law.*?\.',
            'warranty': r'warrant(y|ies).*?\.',
            'payment_terms': r'payment.*?term.*?\.',
            'intellectual_property': r'intellectual property.*?\.'
        }
        
        clauses = []
        for clause_type, pattern in clause_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clauses.append({
                    'type': clause_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return clauses