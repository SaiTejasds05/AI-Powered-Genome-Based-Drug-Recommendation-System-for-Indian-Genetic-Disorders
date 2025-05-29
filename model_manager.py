import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import logging
from datetime import datetime
from config import MODEL_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.metadata_file = Path(self.config['version_control']['path']) / self.config['version_control']['metadata_file']
        self.load_metadata()
    
    def load_metadata(self):
        """Load model metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "models": {},
                "current_version": self.config['current_version'],
                "history": []
            }
            self.save_metadata()
    
    def save_metadata(self):
        """Save model metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def save_model(self, model: torch.nn.Module, model_type: str, version: str, 
                  metrics: Dict[str, float], description: str = "") -> bool:
        """Save a new model version"""
        try:
            # Create version directory
            version_dir = Path(self.config['version_control']['path']) / model_type / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = version_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Update metadata
            model_info = {
                "version": version,
                "type": model_type,
                "path": str(model_path),
                "metrics": metrics,
                "description": description,
                "created_at": datetime.now().isoformat()
            }
            
            self.metadata["models"][f"{model_type}_{version}"] = model_info
            self.metadata["history"].append({
                "action": "save",
                "model_type": model_type,
                "version": version,
                "timestamp": datetime.now().isoformat()
            })
            
            self.save_metadata()
            logger.info(f"Model saved successfully: {model_type} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_type: str, version: Optional[str] = None) -> Optional[torch.nn.Module]:
        """Load a specific model version"""
        try:
            if version is None:
                version = self.config['current_version']
            
            model_key = f"{model_type}_{version}"
            if model_key not in self.metadata["models"]:
                logger.error(f"Model not found: {model_key}")
                return None
            
            model_info = self.metadata["models"][model_key]
            model_path = Path(model_info["path"])
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Load model architecture (this would need to be implemented based on your model types)
            model = self._get_model_architecture(model_type)
            model.load_state_dict(torch.load(model_path))
            
            logger.info(f"Model loaded successfully: {model_type} v{version}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def _get_model_architecture(self, model_type: str) -> torch.nn.Module:
        """Get the appropriate model architecture based on type"""
        # This would need to be implemented based on your specific model architectures
        if model_type == "gnn":
            # Return GNN model architecture
            pass
        elif model_type == "ontology":
            # Return ontology model architecture
            pass
        else:
            # Return standard model architecture
            pass
    
    def get_model_metrics(self, model_type: str, version: str) -> Dict[str, float]:
        """Get metrics for a specific model version"""
        model_key = f"{model_type}_{version}"
        if model_key in self.metadata["models"]:
            return self.metadata["models"][model_key]["metrics"]
        return {}
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models and their versions"""
        return {
            model_type: {
                version: self.metadata["models"][f"{model_type}_{version}"]
                for version in self.config['supported_versions']
                if f"{model_type}_{version}" in self.metadata["models"]
            }
            for model_type in self.config['model_types'].keys()
        }
    
    def compare_versions(self, model_type: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare metrics between two model versions"""
        metrics1 = self.get_model_metrics(model_type, version1)
        metrics2 = self.get_model_metrics(model_type, version2)
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics": {}
        }
        
        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            comparison["metrics"][metric] = {
                "version1": metrics1.get(metric, None),
                "version2": metrics2.get(metric, None),
                "difference": metrics2.get(metric, 0) - metrics1.get(metric, 0)
            }
        
        return comparison

# Initialize model manager
model_manager = ModelManager() 