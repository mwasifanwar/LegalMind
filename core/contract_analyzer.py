import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any
import numpy as np

class ContractAnalyzer:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained('legal-bert')
        self.model.to(self.device)
        self.model.eval()
        
        self.clause_types = [
            'parties', 'term', 'payment', 'confidentiality', 
            'warranties', 'liability', 'termination', 'governing_law', 'ip'
        ]
        
        self.risk_levels = ['low', 'medium', 'high', 'critical']
    
    def _load_model(self, model_path):
        from models.legal_bert import LegalBERT
        model = LegalBERT(num_labels=len(self.clause_types))
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def analyze_contract(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        analysis_results = {
            'overall_risk_score': 0,
            'clause_analysis': [],
            'risk_factors': [],
            'compliance_issues': [],
            'recommendations': []
        }
        
        total_risk = 0
        analyzed_clauses = 0
        
        for segment in document_data['segments']:
            clause_analysis = self._analyze_clause(segment)
            analysis_results['clause_analysis'].append(clause_analysis)
            
            if clause_analysis['risk_level'] != 'low':
                total_risk += self._risk_to_score(clause_analysis['risk_level'])
                analyzed_clauses += 1
        
        if analyzed_clauses > 0:
            analysis_results['overall_risk_score'] = total_risk / analyzed_clauses
        
        analysis_results['risk_factors'] = self._identify_risk_factors(analysis_results['clause_analysis'])
        analysis_results['compliance_issues'] = self._check_compliance(analysis_results['clause_analysis'])
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        
        return analysis_results
    
    def _analyze_clause(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            input_ids = segment['input_ids'].to(self.device)
            attention_mask = segment['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.softmax(outputs, dim=1)
            
            clause_type_idx = torch.argmax(predictions, dim=1).item()
            confidence = torch.max(predictions).item()
            
            clause_type = self.clause_types[clause_type_idx]
            risk_level = self._assess_risk_level(clause_type, segment['text'], confidence)
            
            return {
                'clause_id': segment['id'],
                'clause_type': clause_type,
                'confidence': confidence,
                'risk_level': risk_level,
                'text_snippet': segment['text'][:200] + '...' if len(segment['text']) > 200 else segment['text'],
                'start_char': segment['start_char'],
                'end_char': segment['end_char']
            }
    
    def _assess_risk_level(self, clause_type: str, text: str, confidence: float) -> str:
        risk_keywords = {
            'high': ['unlimited', 'absolute', 'sole discretion', 'without cause', 'immediately'],
            'medium': ['reasonable', 'material breach', 'cure period', '30 days', '60 days'],
            'low': ['mutual', 'good faith', 'written consent', 'prior notice']
        }
        
        text_lower = text.lower()
        
        base_risk = 'low'
        if clause_type in ['liability', 'termination', 'confidentiality']:
            base_risk = 'medium'
        elif clause_type in ['indemnification', 'warranties']:
            base_risk = 'high'
        
        for risk_level, keywords in risk_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if risk_level == 'high':
                    return 'high'
                elif risk_level == 'medium' and base_risk == 'low':
                    base_risk = 'medium'
        
        if confidence < 0.7:
            base_risk = 'medium'
        
        return base_risk
    
    def _risk_to_score(self, risk_level: str) -> int:
        risk_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        return risk_scores.get(risk_level, 1)
    
    def _identify_risk_factors(self, clause_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        risk_factors = []
        
        for clause in clause_analysis:
            if clause['risk_level'] in ['high', 'critical']:
                risk_factors.append({
                    'clause_type': clause['clause_type'],
                    'risk_level': clause['risk_level'],
                    'description': f"High risk {clause['clause_type']} clause detected",
                    'location': f"Segment {clause['clause_id']}",
                    'suggestion': self._get_risk_suggestion(clause['clause_type'])
                })
        
        return risk_factors
    
    def _check_compliance(self, clause_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compliance_issues = []
        
        compliance_rules = {
            'gdpr': ['personal data', 'data processing', 'data subject'],
            'ccpa': ['california', 'consumer privacy'],
            'sox': ['internal control', 'financial reporting'],
            'hippa': ['protected health', 'phi', 'health information']
        }
        
        for clause in clause_analysis:
            text_lower = clause['text_snippet'].lower()
            
            for regulation, keywords in compliance_rules.items():
                if any(keyword in text_lower for keyword in keywords):
                    compliance_issues.append({
                        'regulation': regulation,
                        'clause_type': clause['clause_type'],
                        'description': f"Potential {regulation} compliance issue",
                        'severity': 'medium'
                    })
        
        return compliance_issues
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        risk_score = analysis_results['overall_risk_score']
        if risk_score > 2.5:
            recommendations.append("Consider legal review: High overall risk score detected")
        
        high_risk_clauses = [c for c in analysis_results['clause_analysis'] if c['risk_level'] in ['high', 'critical']]
        if len(high_risk_clauses) > 3:
            recommendations.append("Multiple high-risk clauses identified: Review termination and liability sections")
        
        if analysis_results['compliance_issues']:
            regulations = set(issue['regulation'] for issue in analysis_results['compliance_issues'])
            recommendations.append(f"Compliance review needed for: {', '.join(regulations)}")
        
        if not any(c['clause_type'] == 'governing_law' for c in analysis_results['clause_analysis']):
            recommendations.append("Consider adding governing law clause")
        
        return recommendations
    
    def _get_risk_suggestion(self, clause_type: str) -> str:
        suggestions = {
            'liability': "Consider adding limitation of liability caps",
            'termination': "Define clear cure periods and notice requirements",
            'confidentiality': "Specify confidentiality duration and return obligations",
            'indemnification': "Limit indemnification scope and add procedural requirements",
            'warranties': "Make warranties specific and limited in duration"
        }
        return suggestions.get(clause_type, "Review with legal counsel")
    
    def compare_contracts(self, contract1: Dict[str, Any], contract2: Dict[str, Any]) -> Dict[str, Any]:
        comparison = {
            'similarity_score': 0,
            'key_differences': [],
            'risk_comparison': {}
        }
        
        clauses1 = {c['clause_type']: c for c in contract1['clause_analysis']}
        clauses2 = {c['clause_type']: c for c in contract2['clause_analysis']}
        
        all_clause_types = set(clauses1.keys()) | set(clauses2.keys())
        
        for clause_type in all_clause_types:
            clause1 = clauses1.get(clause_type)
            clause2 = clauses2.get(clause_type)
            
            if clause1 and clause2:
                risk_diff = self._risk_to_score(clause1['risk_level']) - self._risk_to_score(clause2['risk_level'])
                if abs(risk_diff) > 1:
                    comparison['key_differences'].append({
                        'clause_type': clause_type,
                        'contract1_risk': clause1['risk_level'],
                        'contract2_risk': clause2['risk_level'],
                        'difference': 'higher' if risk_diff > 0 else 'lower'
                    })
            elif clause1:
                comparison['key_differences'].append({
                    'clause_type': clause_type,
                    'difference': 'only_in_contract1',
                    'risk_level': clause1['risk_level']
                })
            elif clause2:
                comparison['key_differences'].append({
                    'clause_type': clause_type,
                    'difference': 'only_in_contract2', 
                    'risk_level': clause2['risk_level']
                })
        
        return comparison