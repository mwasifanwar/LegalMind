import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import re
from typing import List, Dict, Any

class RiskDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained('legal-bert')
        self.model.to(self.device)
        self.model.eval()
        
        self.risk_patterns = {
            'unlimited_liability': [
                r'unlimited liability', r'without limitation', r'all damages',
                r'consequential damages', r'indirect damages'
            ],
            'broad_termination': [
                r'terminate.*?without cause', r'immediate termination',
                r'at.*?sole discretion', r'for any reason'
            ],
            'vague_terms': [
                r'reasonable', r'satisfactory', r'commercially reasonable',
                r'material', r'substantial'
            ],
            'one_sided_indemnification': [
                r'indemnify.*?against all', r'hold harmless.*?from any',
                r'defend.*?indemnify'
            ],
            'overly_broad_ip': [
                r'all intellectual property', r'background ip',
                r'pre-existing ip'
            ]
        }
    
    def _load_model(self, model_path):
        from models.compliance_checker import ComplianceChecker
        model = ComplianceChecker()
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def detect_risks(self, text: str, clause_type: str = None) -> List[Dict[str, Any]]:
        risks = []
        
        pattern_risks = self._detect_pattern_risks(text)
        risks.extend(pattern_risks)
        
        semantic_risks = self._detect_semantic_risks(text, clause_type)
        risks.extend(semantic_risks)
        
        compliance_risks = self._detect_compliance_risks(text)
        risks.extend(compliance_risks)
        
        return sorted(risks, key=lambda x: self._risk_priority(x['risk_level']), reverse=True)
    
    def _detect_pattern_risks(self, text: str) -> List[Dict[str, Any]]:
        risks = []
        text_lower = text.lower()
        
        for risk_type, patterns in self.risk_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    risk_level = self._get_pattern_risk_level(risk_type)
                    risks.append({
                        'type': risk_type,
                        'risk_level': risk_level,
                        'description': self._get_risk_description(risk_type),
                        'text_snippet': text[max(0, match.start()-50):match.end()+50],
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'detection_method': 'pattern'
                    })
        
        return risks
    
    def _detect_semantic_risks(self, text: str, clause_type: str = None) -> List[Dict[str, Any]]:
        risks = []
        
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
            risk_scores = torch.softmax(outputs, dim=1)
            
            high_risk_prob = risk_scores[0][1].item()
            
            if high_risk_prob > 0.7:
                risks.append({
                    'type': 'semantic_high_risk',
                    'risk_level': 'high',
                    'description': 'AI-detected high semantic risk',
                    'text_snippet': text[:200] + '...' if len(text) > 200 else text,
                    'confidence': high_risk_prob,
                    'detection_method': 'semantic'
                })
        
        return risks
    
    def _detect_compliance_risks(self, text: str) -> List[Dict[str, Any]]:
        compliance_patterns = {
            'gdpr_risk': [
                r'personal data.*?transfer', r'data processing.*?agreement',
                r'eu.*?data', r'gdpr'
            ],
            'export_control': [
                r'export control', r'ear', r'itar', r'dual use'
            ],
            'anti_corruption': [
                r'anti.?corruption', r'fcpa', r'bribery',
                r'facilitation payment'
            ]
        }
        
        risks = []
        text_lower = text.lower()
        
        for risk_type, patterns in compliance_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    risks.append({
                        'type': risk_type,
                        'risk_level': 'medium',
                        'description': f'Potential {risk_type.upper()} compliance issue',
                        'text_snippet': text[:200] + '...' if len(text) > 200 else text,
                        'detection_method': 'compliance'
                    })
        
        return risks
    
    def _get_pattern_risk_level(self, risk_type: str) -> str:
        risk_levels = {
            'unlimited_liability': 'critical',
            'one_sided_indemnification': 'high', 
            'broad_termination': 'high',
            'overly_broad_ip': 'medium',
            'vague_terms': 'low'
        }
        return risk_levels.get(risk_type, 'medium')
    
    def _get_risk_description(self, risk_type: str) -> str:
        descriptions = {
            'unlimited_liability': 'Unlimited liability exposure detected',
            'broad_termination': 'Overly broad termination rights',
            'vague_terms': 'Vague or subjective language',
            'one_sided_indemnification': 'One-sided indemnification clause',
            'overly_broad_ip': 'Overly broad intellectual property rights'
        }
        return descriptions.get(risk_type, 'Potential risk detected')
    
    def _risk_priority(self, risk_level: str) -> int:
        priorities = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        return priorities.get(risk_level, 1)
    
    def calculate_risk_score(self, risks: List[Dict[str, Any]]) -> float:
        if not risks:
            return 0.0
        
        risk_weights = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        total_weight = sum(risk_weights[r['risk_level']] for r in risks)
        
        max_possible = len(risks) * 4
        return (total_weight / max_possible) * 100 if max_possible > 0 else 0
    
    def generate_risk_report(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        all_risks = []
        
        for segment in document_data['segments']:
            segment_risks = self.detect_risks(segment['text'], segment.get('type'))
            for risk in segment_risks:
                risk['segment_id'] = segment['id']
                risk['clause_type'] = segment.get('type', 'unknown')
                all_risks.append(risk)
        
        overall_score = self.calculate_risk_score(all_risks)
        
        risk_by_type = {}
        for risk in all_risks:
            risk_type = risk['type']
            if risk_type not in risk_by_type:
                risk_by_type[risk_type] = []
            risk_by_type[risk_type].append(risk)
        
        return {
            'overall_risk_score': overall_score,
            'total_risks': len(all_risks),
            'risks_by_severity': self._group_by_severity(all_risks),
            'risks_by_type': risk_by_type,
            'detailed_risks': all_risks,
            'recommendations': self._generate_risk_recommendations(all_risks)
        }
    
    def _group_by_severity(self, risks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {'critical': [], 'high': [], 'medium': [], 'low': []}
        for risk in risks:
            grouped[risk['risk_level']].append(risk)
        return grouped
    
    def _generate_risk_recommendations(self, risks: List[Dict[str, Any]]) -> List[str]:
        recommendations = []
        
        critical_risks = [r for r in risks if r['risk_level'] == 'critical']
        if critical_risks:
            recommendations.append("Immediate legal review required: Critical risks detected")
        
        high_risks = [r for r in risks if r['risk_level'] == 'high']
        if len(high_risks) > 2:
            recommendations.append("Multiple high-risk items: Focus on liability and termination clauses")
        
        vague_terms = [r for r in risks if r['type'] == 'vague_terms']
        if vague_terms:
            recommendations.append("Consider defining vague terms more specifically")
        
        compliance_risks = [r for r in risks if r['detection_method'] == 'compliance']
        if compliance_risks:
            regulations = set(r['type'].split('_')[0].upper() for r in compliance_risks)
            recommendations.append(f"Compliance review needed for: {', '.join(regulations)}")
        
        return recommendations