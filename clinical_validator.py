import streamlit as st
from typing import Dict, List, Any
import json
import pandas as pd

class ClinicalValidator:
    def __init__(self):
        self.load_clinical_guidelines()
    
    def load_clinical_guidelines(self):
        """Load clinical guidelines and drug information"""
        # This would typically load from a database or external source
        self.guidelines = {
            'CYP2D6': {
                'poor_metabolizer': {
                    'avoid': ['codeine', 'tramadol'],
                    'reduce_dose': ['tamoxifen', 'fluoxetine'],
                    'monitor': ['metoprolol', 'carvedilol']
                }
            },
            'CYP2C19': {
                'poor_metabolizer': {
                    'avoid': ['clopidogrel'],
                    'reduce_dose': ['omeprazole', 'pantoprazole'],
                    'monitor': ['citalopram', 'escitalopram']
                }
            },
            'SLCO1B1': {
                'decreased_function': {
                    'reduce_dose': ['simvastatin', 'atorvastatin'],
                    'monitor': ['rosuvastatin']
                }
            }
        }
    
    def validate_recommendations(self, recommendations: List[Dict[str, Any]], 
                              genetic_profile: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate drug recommendations against clinical guidelines"""
        validated_recommendations = []
        
        for rec in recommendations:
            validated_rec = rec.copy()
            
            # Add clinical validation
            gene = rec.get('gene_interaction')
            if gene in self.guidelines:
                gene_guidelines = self.guidelines[gene]
                variant_effect = genetic_profile.get(gene, '')
                
                if variant_effect in gene_guidelines:
                    guidelines = gene_guidelines[variant_effect]
                    
                    # Add clinical warnings
                    if rec['drug_name'].lower() in [d.lower() for d in guidelines.get('avoid', [])]:
                        validated_rec['clinical_warning'] = "AVOID: Contraindicated based on genetic profile"
                        validated_rec['effectiveness_score'] *= 0.3
                    elif rec['drug_name'].lower() in [d.lower() for d in guidelines.get('reduce_dose', [])]:
                        validated_rec['clinical_warning'] = "CAUTION: Consider dose reduction"
                        validated_rec['effectiveness_score'] *= 0.7
                    elif rec['drug_name'].lower() in [d.lower() for d in guidelines.get('monitor', [])]:
                        validated_rec['clinical_warning'] = "MONITOR: Close monitoring recommended"
                        validated_rec['effectiveness_score'] *= 0.9
            
            # Add clinical references
            validated_rec['clinical_references'] = self.get_clinical_references(rec['drug_name'])
            
            validated_recommendations.append(validated_rec)
        
        return validated_recommendations
    
    def get_clinical_references(self, drug_name: str) -> List[Dict[str, str]]:
        """Get clinical references for a drug"""
        # This would typically query a database of clinical studies
        return [
            {
                'title': f"Clinical Study on {drug_name}",
                'journal': "Journal of Clinical Pharmacology",
                'year': "2023",
                'url': "#"
            }
        ]
    
    def get_dosage_recommendations(self, drug_name: str, 
                                 genetic_profile: Dict[str, str]) -> Dict[str, Any]:
        """Get dosage recommendations based on genetic profile"""
        # This would typically use a more sophisticated algorithm
        base_dosage = {
            'simvastatin': '40mg daily',
            'atorvastatin': '20mg daily',
            'clopidogrel': '75mg daily'
        }
        
        if drug_name.lower() in base_dosage:
            dosage = base_dosage[drug_name.lower()]
            return {
                'standard_dosage': dosage,
                'recommended_dosage': self.adjust_dosage(dosage, genetic_profile),
                'monitoring_parameters': self.get_monitoring_parameters(drug_name)
            }
        return None
    
    def adjust_dosage(self, base_dosage: str, genetic_profile: Dict[str, str]) -> str:
        """Adjust dosage based on genetic profile"""
        # This would be a more sophisticated algorithm in practice
        return base_dosage
    
    def get_monitoring_parameters(self, drug_name: str) -> List[str]:
        """Get recommended monitoring parameters for a drug"""
        monitoring_params = {
            'simvastatin': ['Liver function tests', 'CK levels', 'Muscle symptoms'],
            'clopidogrel': ['Platelet function', 'Bleeding time'],
            'warfarin': ['INR', 'Bleeding signs']
        }
        return monitoring_params.get(drug_name.lower(), ['Standard monitoring'])
    
    def display_clinical_disclaimer(self):
        """Display clinical disclaimer"""
        st.warning("""
        ### Clinical Disclaimer
        
        This tool is intended to assist healthcare professionals in making clinical decisions.
        All recommendations should be reviewed by qualified medical personnel.
        
        - Genetic testing results should be interpreted in the context of the patient's complete clinical picture
        - Drug recommendations are based on current pharmacogenetic guidelines
        - Individual patient factors may require adjustments to recommendations
        - Regular monitoring is essential when implementing pharmacogenetic-guided therapy
        
        For more information, please refer to the clinical references provided with each recommendation.
        """)
    
    def display_clinical_references(self):
        """Display clinical references"""
        st.markdown("""
        ### Clinical References
        
        - [PharmGKB Clinical Guidelines](https://www.pharmgkb.org/guideline)
        - [FDA Drug Labeling](https://www.fda.gov/drugs/science-research-drugs/table-pharmacogenomic-biomarkers-drug-labeling)
        - [Clinical Pharmacogenetics Implementation Consortium (CPIC) Guidelines](https://cpicpgx.org/guidelines/)
        """) 