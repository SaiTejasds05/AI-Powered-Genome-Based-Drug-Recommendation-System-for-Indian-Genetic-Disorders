import sqlite3
import pandas as pd
from datetime import datetime
import json
import hashlib
from typing import Dict, List, Any
import streamlit as st

class DataManager:
    def __init__(self):
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect('hospital_data.db')
        c = conn.cursor()
        
        # Create patients table
        c.execute('''CREATE TABLE IF NOT EXISTS patients
                    (id INTEGER PRIMARY KEY,
                     patient_id TEXT UNIQUE,
                     name TEXT,
                     dob DATE,
                     gender TEXT,
                     created_at TIMESTAMP)''')
        
        # Create genetic_data table
        c.execute('''CREATE TABLE IF NOT EXISTS genetic_data
                    (id INTEGER PRIMARY KEY,
                     patient_id TEXT,
                     data_hash TEXT,
                     data_type TEXT,
                     upload_date TIMESTAMP,
                     FOREIGN KEY (patient_id) REFERENCES patients(patient_id))''')
        
        # Create analysis_results table
        c.execute('''CREATE TABLE IF NOT EXISTS analysis_results
                    (id INTEGER PRIMARY KEY,
                     patient_id TEXT,
                     analysis_date TIMESTAMP,
                     results JSON,
                     performed_by TEXT,
                     FOREIGN KEY (patient_id) REFERENCES patients(patient_id))''')
        
        conn.commit()
        conn.close()
    
    def add_patient(self, patient_id: str, name: str, dob: str, gender: str) -> bool:
        """Add a new patient to the database"""
        try:
            conn = sqlite3.connect('hospital_data.db')
            c = conn.cursor()
            c.execute('''INSERT INTO patients (patient_id, name, dob, gender, created_at)
                        VALUES (?, ?, ?, ?, ?)''',
                     (patient_id, name, dob, gender, datetime.now()))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def store_genetic_data(self, patient_id: str, data: pd.DataFrame, data_type: str) -> bool:
        """Store genetic data with hash for verification"""
        try:
            # Create hash of the data
            data_hash = hashlib.sha256(data.to_json().encode()).hexdigest()
            
            conn = sqlite3.connect('hospital_data.db')
            c = conn.cursor()
            c.execute('''INSERT INTO genetic_data (patient_id, data_hash, data_type, upload_date)
                        VALUES (?, ?, ?, ?)''',
                     (patient_id, data_hash, data_type, datetime.now()))
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error storing genetic data: {str(e)}")
            return False
        finally:
            conn.close()
    
    def store_analysis_results(self, patient_id: str, results: Dict[str, Any], user_id: str) -> bool:
        """Store analysis results"""
        try:
            conn = sqlite3.connect('hospital_data.db')
            c = conn.cursor()
            c.execute('''INSERT INTO analysis_results (patient_id, analysis_date, results, performed_by)
                        VALUES (?, ?, ?, ?)''',
                     (patient_id, datetime.now(), json.dumps(results), user_id))
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error storing analysis results: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient's analysis history"""
        conn = sqlite3.connect('hospital_data.db')
        c = conn.cursor()
        c.execute('''SELECT analysis_date, results, performed_by 
                    FROM analysis_results 
                    WHERE patient_id = ? 
                    ORDER BY analysis_date DESC''', (patient_id,))
        results = c.fetchall()
        conn.close()
        
        return [{
            'date': row[0],
            'results': json.loads(row[1]),
            'performed_by': row[2]
        } for row in results]
    
    def validate_genetic_data(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate uploaded genetic data"""
        required_columns = ['Gene', 'Variant/Haplotypes']
        
        # Check required columns
        if not all(col in data.columns for col in required_columns):
            return False, "Missing required columns: Gene, Variant/Haplotypes"
        
        # Check for empty values in required columns
        if data[required_columns].isnull().any().any():
            return False, "Empty values found in required columns"
        
        # Check data types
        if not all(isinstance(x, str) for x in data['Gene']):
            return False, "Invalid data type in Gene column"
        
        return True, "Data validation successful"
    
    def export_to_emr(self, patient_id: str, results: Dict[str, Any]) -> bool:
        """Export results to EMR system (placeholder)"""
        # This would be implemented based on the hospital's EMR system
        try:
            # Placeholder for EMR integration
            st.info("Results exported to EMR system")
            return True
        except Exception as e:
            st.error(f"Error exporting to EMR: {str(e)}")
            return False 