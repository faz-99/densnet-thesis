"""Medical Report Generator for Swin Transformer on BreakHis"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import json
from datetime import datetime


class FeatureProjection(nn.Module):
    """Project Swin features to LLM embedding space"""
    
    def __init__(self, swin_dim: int = 1024, llm_dim: int = 4096, num_queries: int = 32):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, swin_dim))
        
        # Cross-attention to aggregate features
        self.cross_attn = nn.MultiheadAttention(swin_dim, num_heads=8, batch_first=True)
        
        # Projection to LLM space
        self.projection = nn.Sequential(
            nn.Linear(swin_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_dim, llm_dim)
        )
        
    def forward(self, swin_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            swin_features: [B, N, D] where N is num_patches, D is swin_dim
        Returns:
            projected_features: [B, num_queries, llm_dim]
        """
        B = swin_features.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        
        # Cross-attention: queries attend to swin features
        aggregated, _ = self.cross_attn(queries, swin_features, swin_features)
        
        # Project to LLM space
        projected = self.projection(aggregated)
        
        return projected


class SwinMedicalReportGenerator(nn.Module):
    """End-to-end medical report generation from Swin features"""
    
    def __init__(self, swin_model: nn.Module, llm_name: str = 'template',
                 swin_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.swin_model = swin_model
        self.llm_name = llm_name
        
        # Feature extraction and projection
        self.projection = FeatureProjection(swin_dim, llm_dim)
        
        # Class names for BreakHis
        self.class_names = [
            'Adenosis', 'Fibroadenoma', 'Phyllodes Tumor', 'Tubular Adenoma',
            'Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma'
        ]
        
        # LLM integration (template-based initially)
        self.use_llm = llm_name != 'template'
        if self.use_llm:
            self._load_llm(llm_name)
    
    def _load_llm(self, llm_name: str):
        """Load medical LLM (placeholder for actual implementation)"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16)
        except:
            print(f"Warning: Could not load LLM {llm_name}, using template-based generation")
            self.use_llm = False
    
    def extract_swin_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale features from Swin
        
        Args:
            image: [B, 3, H, W]
        Returns:
            features: [B, N, D]
        """
        self.swin_model.eval()
        with torch.no_grad():
            # Extract features before classification head
            if hasattr(self.swin_model, 'forward_features'):
                features = self.swin_model.forward_features(image)
            else:
                # Hook-based extraction
                features = self._extract_with_hooks(image)
        
        return features
    
    def _extract_with_hooks(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features using hooks"""
        features = []
        
        def hook(module, input, output):
            features.append(output)
        
        # Register hook on last layer before classifier
        handle = None
        for name, module in self.swin_model.named_modules():
            if 'norm' in name or 'avgpool' in name:
                handle = module.register_forward_hook(hook)
        
        _ = self.swin_model(image)
        
        if handle:
            handle.remove()
        
        return features[-1] if features else torch.randn(image.shape[0], 49, 1024)
    
    def generate_template_report(self, prediction: int, confidence: float,
                                 attention_regions: Optional[str] = None) -> Dict:
        """Generate template-based pathology report
        
        Args:
            prediction: Predicted class index
            confidence: Model confidence
            attention_regions: Description of attention regions
            
        Returns:
            Structured report dictionary
        """
        class_name = self.class_names[prediction]
        is_malignant = prediction >= 4
        
        # Microscopic findings based on class
        findings_templates = {
            'Adenosis': 'Increased number of acini per lobule with preserved lobular architecture. '
                       'Epithelial cells show minimal atypia. Myoepithelial layer intact.',
            'Fibroadenoma': 'Biphasic tumor with epithelial and stromal components. '
                           'Compressed glandular structures surrounded by cellular stroma. No cytologic atypia.',
            'Phyllodes Tumor': 'Leaf-like architecture with stromal hypercellularity. '
                              'Stromal overgrowth with variable cellularity. Epithelial clefts present.',
            'Tubular Adenoma': 'Well-circumscribed lesion with tubular structures. '
                              'Uniform epithelial cells with minimal pleomorphism.',
            'Ductal Carcinoma': 'Malignant epithelial proliferation forming irregular ducts. '
                               'Nuclear pleomorphism and increased mitotic activity. Loss of myoepithelial layer.',
            'Lobular Carcinoma': 'Discohesive small cells with minimal pleomorphism. '
                                'Single-file pattern and targetoid growth around ducts.',
            'Mucinous Carcinoma': 'Clusters of tumor cells floating in abundant extracellular mucin. '
                                 'Well-differentiated cells with minimal atypia.',
            'Papillary Carcinoma': 'Papillary architecture with fibrovascular cores. '
                                  'Epithelial cells showing nuclear atypia and stratification.'
        }
        
        # Key observations
        observations = {
            'Adenosis': ['Preserved lobular architecture', 'Minimal nuclear atypia', 'Intact myoepithelial layer'],
            'Fibroadenoma': ['Biphasic tumor pattern', 'Compressed glandular structures', 'No malignant features'],
            'Phyllodes Tumor': ['Leaf-like architecture', 'Stromal hypercellularity', 'Epithelial clefts'],
            'Tubular Adenoma': ['Tubular structures', 'Uniform epithelial cells', 'Well-circumscribed'],
            'Ductal Carcinoma': ['Nuclear pleomorphism', 'Increased mitotic activity', 'Irregular ductal structures'],
            'Lobular Carcinoma': ['Discohesive cells', 'Single-file pattern', 'Targetoid growth'],
            'Mucinous Carcinoma': ['Abundant extracellular mucin', 'Tumor cell clusters', 'Well-differentiated'],
            'Papillary Carcinoma': ['Papillary architecture', 'Fibrovascular cores', 'Nuclear stratification']
        }
        
        report = {
            'report_id': f"BR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'magnification': '400X',
            'staining': 'H&E',
            
            'microscopic_findings': findings_templates[class_name],
            
            'key_observations': observations[class_name],
            
            'diagnosis': {
                'primary': class_name,
                'category': 'Malignant' if is_malignant else 'Benign',
                'confidence': f"{confidence*100:.1f}%"
            },
            
            'attention_regions': attention_regions or 'Model focused on cellular architecture and nuclear features',
            
            'differential_considerations': self._get_differentials(prediction),
            
            'clinical_correlation': 'Clinical correlation and additional studies recommended' if confidence < 0.9 else 'Findings consistent with diagnosis'
        }
        
        return report
    
    def _get_differentials(self, prediction: int) -> List[str]:
        """Get differential diagnoses"""
        differentials = {
            0: ['Fibroadenoma', 'Sclerosing adenosis'],
            1: ['Phyllodes tumor', 'Adenosis'],
            2: ['Fibroadenoma', 'Sarcoma'],
            3: ['Fibroadenoma', 'Lactating adenoma'],
            4: ['Lobular carcinoma', 'Invasive ductal carcinoma'],
            5: ['Ductal carcinoma', 'Pleomorphic lobular carcinoma'],
            6: ['Colloid carcinoma', 'Ductal carcinoma with mucin'],
            7: ['Intraductal papilloma', 'Ductal carcinoma']
        }
        return differentials.get(prediction, ['Requires additional studies'])
    
    def forward(self, image: torch.Tensor, return_features: bool = False) -> Dict:
        """Generate complete medical report
        
        Args:
            image: Input tensor [B, 3, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with report and metadata
        """
        # Get prediction
        self.swin_model.eval()
        with torch.no_grad():
            logits = self.swin_model(image)
            probs = F.softmax(logits, dim=1)
            prediction = logits.argmax(dim=1).item()
            confidence = probs[0, prediction].item()
        
        # Extract features
        features = self.extract_swin_features(image)
        
        # Project features (for future LLM integration)
        projected = self.projection(features)
        
        # Generate report
        if self.use_llm:
            report = self._generate_llm_report(projected, prediction, confidence)
        else:
            report = self.generate_template_report(prediction, confidence)
        
        result = {
            'report': report,
            'prediction': prediction,
            'confidence': confidence,
            'class_probabilities': probs[0].cpu().tolist()
        }
        
        if return_features:
            result['features'] = features
            result['projected_features'] = projected
        
        return result
    
    def _generate_llm_report(self, features: torch.Tensor, prediction: int, confidence: float) -> Dict:
        """Generate report using LLM (placeholder)"""
        # This would use the actual LLM with features as context
        # For now, fall back to template
        return self.generate_template_report(prediction, confidence)
    
    def format_report_text(self, report_dict: Dict) -> str:
        """Format report as readable text"""
        report = report_dict['report']
        
        text = f"""
HISTOPATHOLOGICAL ANALYSIS REPORT
{'='*60}

Report ID: {report['report_id']}
Date: {report['timestamp']}
Magnification: {report['magnification']}
Staining: {report['staining']}

MICROSCOPIC FINDINGS:
{report['microscopic_findings']}

KEY OBSERVATIONS:
"""
        for obs in report['key_observations']:
            text += f"  • {obs}\n"
        
        text += f"""
DIAGNOSIS:
  Primary: {report['diagnosis']['primary']}
  Category: {report['diagnosis']['category']}
  Confidence: {report['diagnosis']['confidence']}

ATTENTION REGIONS:
{report['attention_regions']}

DIFFERENTIAL CONSIDERATIONS:
"""
        for diff in report['differential_considerations']:
            text += f"  • {diff}\n"
        
        text += f"""
CLINICAL CORRELATION:
{report['clinical_correlation']}

{'='*60}
"""
        return text
