from flask import Flask, request, jsonify, send_file
import os
import sys
from datetime import datetime

sys.path.append('..')
from core.document_processor import DocumentProcessor
from core.contract_analyzer import ContractAnalyzer
from core.risk_detector import RiskDetector
from core.summary_generator import SummaryGenerator
from utils.helpers import setup_logging

app = Flask(__name__)
logger = setup_logging()

document_processor = None
contract_analyzer = None
risk_detector = None
summary_generator = None

def initialize_services():
    global document_processor, contract_analyzer, risk_detector, summary_generator
    
    try:
        document_processor = DocumentProcessor()
        contract_analyzer = ContractAnalyzer('./models/legal_bert.pth')
        risk_detector = RiskDetector('./models/compliance_checker.pth')
        summary_generator = SummaryGenerator()
        
        logger.info("All LegalMind services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'LegalMind API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze/contract', methods=['POST'])
def analyze_contract():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No contract file provided'}), 400
        
        contract_file = request.files['file']
        analysis_type = request.form.get('analysis_type', 'full')
        
        file_path = f'./temp_contract_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        contract_file.save(file_path)
        
        if document_processor is None:
            initialize_services()
        
        processed_doc = document_processor.preprocess_contract(file_path)
        
        if analysis_type in ['basic', 'full']:
            analysis_results = contract_analyzer.analyze_contract(processed_doc)
        else:
            analysis_results = {}
        
        if analysis_type in ['risk', 'full']:
            risk_report = risk_detector.generate_risk_report(processed_doc)
        else:
            risk_report = {}
        
        if analysis_type in ['summary', 'full']:
            summary = summary_generator.generate_summary(processed_doc['raw_text'], 'executive')
        else:
            summary = ""
        
        os.remove(file_path)
        
        result = {
            'analysis_type': analysis_type,
            'document_metadata': processed_doc['metadata'],
            'analysis_results': analysis_results,
            'risk_report': risk_report,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        analysis_type = data.get('analysis_type', 'full')
        
        if document_processor is None:
            initialize_services()
        
        processed_doc = {
            'raw_text': text,
            'segments': document_processor.segment_document(text),
            'metadata': {'file_path': 'direct_text_input'}
        }
        
        analysis_results = contract_analyzer.analyze_contract(processed_doc)
        risk_report = risk_detector.generate_risk_report(processed_doc)
        summary = summary_generator.generate_summary(text, 'executive')
        
        result = {
            'analysis_type': analysis_type,
            'analysis_results': analysis_results,
            'risk_report': risk_report,
            'summary': summary
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summary/generate', methods=['POST'])
def generate_summary():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        summary_type = data.get('summary_type', 'executive')
        
        if summary_generator is None:
            initialize_services()
        
        summary = summary_generator.generate_summary(text, summary_type)
        
        return jsonify({
            'summary_type': summary_type,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/risk/assess', methods=['POST'])
def assess_risk():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if risk_detector is None:
            initialize_services()
        
        risks = risk_detector.detect_risks(text)
        risk_score = risk_detector.calculate_risk_score(risks)
        
        return jsonify({
            'risk_score': risk_score,
            'total_risks': len(risks),
            'risks': risks
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compliance/check', methods=['POST'])
def check_compliance():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        regulation = data.get('regulation', 'all')
        
        if risk_detector is None:
            initialize_services()
        
        compliance_risks = risk_detector._detect_compliance_risks(text)
        
        if regulation != 'all':
            compliance_risks = [r for r in compliance_risks if r['type'].startswith(regulation)]
        
        return jsonify({
            'regulation': regulation,
            'compliance_issues': compliance_risks
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare/contracts', methods=['POST'])
def compare_contracts():
    try:
        data = request.get_json()
        
        if not data or 'contract1' not in data or 'contract2' not in data:
            return jsonify({'error': 'Both contracts required'}), 400
        
        contract1_data = data['contract1']
        contract2_data = data['contract2']
        
        if contract_analyzer is None:
            initialize_services()
        
        comparison = contract_analyzer.compare_contracts(contract1_data, contract2_data)
        comparison_summary = summary_generator.generate_comparison_summary(contract1_data, contract2_data)
        
        return jsonify({
            'comparison': comparison,
            'summary': comparison_summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_services()
    app.run(host='0.0.0.0', port=8000, debug=True)