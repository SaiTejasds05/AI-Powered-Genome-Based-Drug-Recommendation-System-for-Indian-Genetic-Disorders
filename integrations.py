import requests
import json
from typing import Dict, Any, Optional
from config import EMR_CONFIG, LAB_CONFIG, PHARMACY_CONFIG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EMRIntegration:
    def __init__(self):
        self.config = EMR_CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        })
    
    def get_patient_info(self, patient_id: str) -> Dict[str, Any]:
        """Get patient information from EMR system"""
        try:
            response = self.session.get(
                f"{self.config['api_url']}{self.config['endpoints']['patient']}/{patient_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching patient info: {str(e)}")
            return {}
    
    def send_analysis_results(self, patient_id: str, results: Dict[str, Any]) -> bool:
        """Send analysis results to EMR system"""
        try:
            response = self.session.post(
                f"{self.config['api_url']}{self.config['endpoints']['results']}",
                json={
                    "patient_id": patient_id,
                    "results": results,
                    "system_id": self.config['system_id']
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error sending results to EMR: {str(e)}")
            return False

class LabIntegration:
    def __init__(self):
        self.config = LAB_CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        })
    
    def get_lab_results(self, patient_id: str) -> Dict[str, Any]:
        """Get laboratory test results"""
        try:
            response = self.session.get(
                f"{self.config['api_url']}{self.config['endpoints']['results']}",
                params={"patient_id": patient_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching lab results: {str(e)}")
            return {}
    
    def order_genetic_test(self, patient_id: str, test_type: str) -> bool:
        """Order genetic test through laboratory system"""
        try:
            response = self.session.post(
                f"{self.config['api_url']}{self.config['endpoints']['orders']}",
                json={
                    "patient_id": patient_id,
                    "test_type": test_type,
                    "priority": "routine"
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error ordering genetic test: {str(e)}")
            return False

class PharmacyIntegration:
    def __init__(self):
        self.config = PHARMACY_CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        })
    
    def check_medication_availability(self, medication: str) -> Dict[str, Any]:
        """Check medication availability in pharmacy"""
        try:
            response = self.session.get(
                f"{self.config['api_url']}{self.config['endpoints']['inventory']}",
                params={"medication": medication}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error checking medication availability: {str(e)}")
            return {}
    
    def send_prescription(self, patient_id: str, prescription: Dict[str, Any]) -> bool:
        """Send prescription to pharmacy system"""
        try:
            response = self.session.post(
                f"{self.config['api_url']}{self.config['endpoints']['prescriptions']}",
                json={
                    "patient_id": patient_id,
                    "prescription": prescription
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error sending prescription: {str(e)}")
            return False

# Initialize integration instances
emr = EMRIntegration()
lab = LabIntegration()
pharmacy = PharmacyIntegration() 