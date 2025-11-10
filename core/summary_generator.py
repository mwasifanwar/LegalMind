import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any

class SummaryGenerator:
    def __init__(self, model_name="legal-t5"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.max_input_length = 1024
        self.max_output_length = 150
    
    def generate_summary(self, text: str, summary_type: str = "executive") -> str:
        if summary_type == "executive":
            return self._generate_executive_summary(text)
        elif summary_type == "detailed":
            return self._generate_detailed_summary(text)
        elif summary_type == "risk_focused":
            return self._generate_risk_summary(text)
        else:
            return self._generate_executive_summary(text)
    
    def _generate_executive_summary(self, text: str) -> str:
        prompt = f"Generate a concise executive summary of this legal contract:\n\n{text}"
        
        summary = self._generate_with_model(prompt)
        return summary or "Unable to generate summary for this document."
    
    def _generate_detailed_summary(self, text: str) -> str:
        sections = self._extract_key_sections(text)
        
        detailed_summary = "CONTRACT SUMMARY\n\n"
        
        for section_name, section_text in sections.items():
            if len(section_text) > 100:
                section_summary = self._summarize_section(section_text, section_name)
                detailed_summary += f"{section_name.upper()}:\n{section_summary}\n\n"
        
        return detailed_summary
    
    def _generate_risk_summary(self, text: str) -> str:
        from .risk_detector import RiskDetector
        
        risk_detector = RiskDetector()
        risks = risk_detector.detect_risks(text)
        
        risk_summary = "RISK ASSESSMENT SUMMARY\n\n"
        
        if not risks:
            risk_summary += "No significant risks detected.\n"
            return risk_summary
        
        risk_by_level = {}
        for risk in risks:
            level = risk['risk_level']
            if level not in risk_by_level:
                risk_by_level[level] = []
            risk_by_level[level].append(risk)
        
        for level in ['critical', 'high', 'medium', 'low']:
            if level in risk_by_level:
                risk_summary += f"{level.upper()} RISKS ({len(risk_by_level[level])}):\n"
                for risk in risk_by_level[level][:3]:
                    risk_summary += f"- {risk['description']}\n"
                risk_summary += "\n"
        
        return risk_summary
    
    def _generate_with_model(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def _extract_key_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        
        section_keywords = {
            'parties': ['party', 'between', 'agreement between'],
            'term': ['term', 'duration', 'effective date', 'expiration'],
            'payment': ['payment', 'fee', 'compensation', 'price', 'consideration'],
            'confidentiality': ['confidential', 'proprietary', 'non-disclosure'],
            'warranties': ['warrant', 'represent', 'covenant'],
            'liability': ['liability', 'indemnif', 'hold harmless', 'damages'],
            'termination': ['terminat', 'breach', 'default', 'cure'],
            'governing_law': ['governing law', 'jurisdiction', 'venue', 'dispute']
        }
        
        text_lower = text.lower()
        lines = text.split('\n')
        
        for section_name, keywords in section_keywords.items():
            section_text = ""
            in_section = False
            
            for line in lines:
                line_lower = line.lower()
                
                if any(keyword in line_lower for keyword in keywords):
                    in_section = True
                
                if in_section:
                    section_text += line + "\n"
                    
                    if len(section_text) > 1000:
                        break
            
            if section_text:
                sections[section_name] = section_text
        
        return sections
    
    def _summarize_section(self, section_text: str, section_name: str) -> str:
        prompt = f"Summarize the {section_name} section of this legal contract:\n\n{section_text}"
        
        summary = self._generate_with_model(prompt)
        return summary or f"Key provisions related to {section_name}."
    
    def generate_comparison_summary(self, contract1: Dict[str, Any], contract2: Dict[str, Any]) -> str:
        from .contract_analyzer import ContractAnalyzer
        
        analyzer = ContractAnalyzer()
        comparison = analyzer.compare_contracts(contract1, contract2)
        
        summary = "CONTRACT COMPARISON SUMMARY\n\n"
        
        summary += f"Overall Similarity: {comparison['similarity_score']:.2f}/10\n\n"
        
        if comparison['key_differences']:
            summary += "KEY DIFFERENCES:\n"
            for diff in comparison['key_differences'][:5]:
                if 'difference' in diff:
                    if diff['difference'] in ['higher', 'lower']:
                        summary += f"- {diff['clause_type']}: Risk is {diff['difference']} in Contract 2\n"
                    else:
                        summary += f"- {diff['clause_type']}: {diff['difference'].replace('_', ' ').title()}\n"
        
        risk1 = contract1.get('analysis_results', {}).get('overall_risk_score', 0)
        risk2 = contract2.get('analysis_results', {}).get('overall_risk_score', 0)
        
        summary += f"\nRISK COMPARISON:\n"
        summary += f"Contract 1 Overall Risk: {risk1:.2f}/4\n"
        summary += f"Contract 2 Overall Risk: {risk2:.2f}/4\n"
        
        if risk1 > risk2:
            summary += "Contract 1 has higher overall risk.\n"
        elif risk2 > risk1:
            summary += "Contract 2 has higher overall risk.\n"
        else:
            summary += "Both contracts have similar risk levels.\n"
        
        return summary
    
    def generate_action_items(self, analysis_results: Dict[str, Any]) -> List[str]:
        action_items = []
        
        risk_score = analysis_results.get('overall_risk_score', 0)
        if risk_score > 2.5:
            action_items.append("Schedule legal review session")
        
        high_risk_factors = [r for r in analysis_results.get('risk_factors', []) 
                           if r['risk_level'] in ['high', 'critical']]
        if high_risk_factors:
            action_items.append("Negotiate high-risk clauses identified in analysis")
        
        compliance_issues = analysis_results.get('compliance_issues', [])
        if compliance_issues:
            regulations = set(issue['regulation'] for issue in compliance_issues)
            action_items.append(f"Verify compliance with: {', '.join(regulations)}")
        
        recommendations = analysis_results.get('recommendations', [])
        for rec in recommendations[:3]:
            action_items.append(rec)
        
        return action_items