import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any

class VisualizationEngine:
    def __init__(self):
        self.color_scheme = {
            'critical': '#FF6B6B',
            'high': '#FFA726', 
            'medium': '#FFE082',
            'low': '#C8E6C9',
            'safe': '#4CAF50'
        }
    
    def create_risk_dashboard(self, analysis_results: Dict[str, Any]) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Distribution', 'Clause Types Analysis', 
                          'Compliance Status', 'Risk Timeline'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "scatter"}]]
        )
        
        risk_data = analysis_results.get('risk_report', {}).get('risks_by_severity', {})
        if risk_data:
            risk_counts = [len(risk_data.get(level, [])) for level in ['critical', 'high', 'medium', 'low']]
            risk_labels = ['Critical', 'High', 'Medium', 'Low']
            
            fig.add_trace(
                go.Pie(
                    labels=risk_labels,
                    values=risk_counts,
                    hole=0.4,
                    marker_colors=[self.color_scheme[level] for level in ['critical', 'high', 'medium', 'low']]
                ),
                row=1, col=1
            )
        
        clause_analysis = analysis_results.get('analysis_results', {}).get('clause_analysis', [])
        if clause_analysis:
            clause_types = {}
            for clause in clause_analysis:
                clause_type = clause['clause_type']
                if clause_type not in clause_types:
                    clause_types[clause_type] = 0
                clause_types[clause_type] += 1
            
            fig.add_trace(
                go.Bar(
                    x=list(clause_types.keys()),
                    y=list(clause_types.values())
                ),
                row=1, col=2
            )
        
        overall_risk = analysis_results.get('analysis_results', {}).get('overall_risk_score', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_risk,
                title={'text': "Overall Risk Score"},
                domain={'row': 2, 'column': 1},
                gauge={
                    'axis': {'range': [0, 4]},
                    'bar': {'color': self._get_risk_color(overall_risk)},
                    'steps': [
                        {'range': [0, 1], 'color': self.color_scheme['low']},
                        {'range': [1, 2], 'color': self.color_scheme['medium']},
                        {'range': [2, 3], 'color': self.color_scheme['high']},
                        {'range': [3, 4], 'color': self.color_scheme['critical']}
                    ]
                }
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Legal Contract Risk Dashboard")
        return fig
    
    def create_compliance_chart(self, compliance_issues: List[Dict[str, Any]]) -> go.Figure:
        if not compliance_issues:
            fig = go.Figure()
            fig.add_annotation(text="No Compliance Issues Detected", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(compliance_issues)
        regulation_counts = df['regulation'].value_counts()
        
        fig = px.bar(
            x=regulation_counts.index,
            y=regulation_counts.values,
            title="Compliance Issues by Regulation",
            labels={'x': 'Regulation', 'y': 'Number of Issues'}
        )
        
        return fig
    
    def create_clause_risk_heatmap(self, clause_analysis: List[Dict[str, Any]]) -> go.Figure:
        risk_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        
        data = []
        for clause in clause_analysis:
            data.append({
                'clause_type': clause['clause_type'],
                'risk_score': risk_scores.get(clause['risk_level'], 1),
                'confidence': clause['confidence']
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No Clause Data Available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        pivot_table = df.pivot_table(
            values='risk_score', 
            index='clause_type', 
            aggfunc='mean'
        ).reset_index()
        
        fig = px.scatter(
            df,
            x='clause_type',
            y='risk_score',
            size='confidence',
            color='risk_score',
            title="Clause Risk Analysis",
            color_continuous_scale='RdYlGn_r'
        )
        
        return fig
    
    def generate_risk_timeline(self, document_segments: List[Dict[str, Any]]) -> go.Figure:
        risk_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        
        positions = []
        risks = []
        texts = []
        
        for segment in document_segments:
            positions.append(segment['start_char'])
            risk_level = segment.get('risk_level', 'low')
            risks.append(risk_scores.get(risk_level, 1))
            texts.append(segment['text'][:100] + '...')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=positions,
            y=risks,
            mode='lines+markers',
            text=texts,
            hovertemplate='<b>Position: %{x}</b><br>Risk: %{y}<br>Text: %{text}<extra></extra>',
            line=dict(color='red', width=2),
            marker=dict(size=8, color=risks, colorscale='RdYlGn_r')
        ))
        
        fig.update_layout(
            title="Risk Distribution Throughout Document",
            xaxis_title="Document Position",
            yaxis_title="Risk Level",
            yaxis=dict(tickvals=[1, 2, 3, 4], ticktext=['Low', 'Medium', 'High', 'Critical'])
        )
        
        return fig
    
    def _get_risk_color(self, risk_score: float) -> str:
        if risk_score >= 3:
            return self.color_scheme['critical']
        elif risk_score >= 2:
            return self.color_scheme['high']
        elif risk_score >= 1:
            return self.color_scheme['medium']
        else:
            return self.color_scheme['low']
    
    def create_comparison_chart(self, contract1: Dict[str, Any], contract2: Dict[str, Any]) -> go.Figure:
        risk1 = contract1.get('analysis_results', {}).get('overall_risk_score', 0)
        risk2 = contract2.get('analysis_results', {}).get('overall_risk_score', 0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Contract 1', 'Contract 2'],
            y=[risk1, risk2],
            marker_color=[self._get_risk_color(risk1), self._get_risk_color(risk2)]
        ))
        
        fig.update_layout(
            title="Contract Risk Comparison",
            yaxis_title="Overall Risk Score",
            yaxis=dict(range=[0, 4])
        )
        
        return fig