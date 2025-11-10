import argparse
import torch
import logging
import os

from core.document_processor import DocumentProcessor
from core.contract_analyzer import ContractAnalyzer
from core.risk_detector import RiskDetector
from core.summary_generator import SummaryGenerator
from utils.config import Config
from utils.helpers import setup_logging, create_directories
from api.server import app

class LegalMind:
    def __init__(self, config_path='config/model_config.yaml'):
        self.config = Config(config_path)
        self.logger = setup_logging()
        create_directories()
        
        self.document_processor = None
        self.contract_analyzer = None
        self.risk_detector = None
        self.summary_generator = None
        
    def initialize_services(self):
        self.logger.info("Initializing LegalMind services...")
        
        self.document_processor = DocumentProcessor()
        self.contract_analyzer = ContractAnalyzer()
        self.risk_detector = RiskDetector()
        self.summary_generator = SummaryGenerator()
        
        self.logger.info("All services initialized successfully")
    
    def analyze_contract(self, file_path, analysis_type="full"):
        self.logger.info(f"Analyzing contract: {file_path}")
        
        if self.document_processor is None:
            self.initialize_services()
        
        processed_doc = self.document_processor.preprocess_contract(file_path)
        
        if analysis_type in ["basic", "full"]:
            analysis_results = self.contract_analyzer.analyze_contract(processed_doc)
        else:
            analysis_results = {}
        
        if analysis_type in ["risk", "full"]:
            risk_report = self.risk_detector.generate_risk_report(processed_doc)
        else:
            risk_report = {}
        
        if analysis_type in ["summary", "full"]:
            summary = self.summary_generator.generate_summary(processed_doc['raw_text'], 'executive')
        else:
            summary = ""
        
        result = {
            'file_path': file_path,
            'analysis_type': analysis_type,
            'document_metadata': processed_doc['metadata'],
            'analysis_results': analysis_results,
            'risk_report': risk_report,
            'summary': summary
        }
        
        return result
    
    def train_models(self):
        self.logger.info("Training LegalMind models...")
        
        from models.legal_bert import LegalBERT
        from data.dataloader import LegalDataLoader
        
        model = LegalBERT()
        data_loader = LegalDataLoader(self.config)
        
        train_loader = data_loader.create_data_loader('./data/train')
        val_loader = data_loader.create_data_loader('./data/val')
        
        from training.trainers import LegalBERTTrainer
        trainer = LegalBERTTrainer(model, train_loader, val_loader, self.config)
        best_accuracy = trainer.train()
        
        self.logger.info(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")
    
    def run_api(self):
        self.logger.info("Starting LegalMind API...")
        app.run(
            host=self.config.get('api.host', 'localhost'),
            port=self.config.get('api.port', 8000),
            debug=self.config.get('api.debug', True)
        )

def main():
    parser = argparse.ArgumentParser(description='LegalMind: AI Contract Analysis & Compliance')
    parser.add_argument('--mode', choices=['analyze', 'train', 'api', 'summary'], 
                       default='analyze', help='Operation mode')
    parser.add_argument('--file', type=str, help='Contract file path for analysis')
    parser.add_argument('--analysis-type', type=str, default='full', 
                       choices=['basic', 'risk', 'summary', 'full'],
                       help='Type of analysis to perform')
    parser.add_argument('--config', type=str, default='config/model_config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    legalmind = LegalMind(args.config)
    
    if args.mode == 'analyze':
        if not args.file:
            print("Please provide a contract file with --file")
            return
        
        result = legalmind.analyze_contract(args.file, args.analysis_type)
        print(f"Analysis completed for {args.file}")
        print(f"Overall Risk Score: {result['analysis_results'].get('overall_risk_score', 0):.2f}")
        
    elif args.mode == 'train':
        legalmind.train_models()
        
    elif args.mode == 'api':
        legalmind.run_api()
        
    elif args.mode == 'summary':
        if not args.file:
            print("Please provide a contract file with --file")
            return
        
        result = legalmind.analyze_contract(args.file, 'summary')
        print("CONTRACT SUMMARY:")
        print(result['summary'])

if __name__ == '__main__':
    main()