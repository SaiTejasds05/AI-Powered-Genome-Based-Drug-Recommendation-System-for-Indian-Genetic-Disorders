import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CERT_DIR = BASE_DIR / "certificates"

# HTTPS Configuration
SSL_CERTIFICATE = CERT_DIR / "cert.pem"
SSL_KEY = CERT_DIR / "key.pem"

# EMR System Integration
EMR_CONFIG = {
    "api_url": os.getenv("EMR_API_URL", "https://emr-api.example.com"),
    "api_key": os.getenv("EMR_API_KEY", ""),
    "system_id": os.getenv("EMR_SYSTEM_ID", ""),
    "endpoints": {
        "patient": "/api/v1/patients",
        "orders": "/api/v1/orders",
        "results": "/api/v1/results"
    }
}

# Laboratory System Integration
LAB_CONFIG = {
    "api_url": os.getenv("LAB_API_URL", "https://lab-api.example.com"),
    "api_key": os.getenv("LAB_API_KEY", ""),
    "endpoints": {
        "tests": "/api/v1/tests",
        "results": "/api/v1/results",
        "orders": "/api/v1/orders"
    }
}

# Pharmacy System Integration
PHARMACY_CONFIG = {
    "api_url": os.getenv("PHARMACY_API_URL", "https://pharmacy-api.example.com"),
    "api_key": os.getenv("PHARMACY_API_KEY", ""),
    "endpoints": {
        "medications": "/api/v1/medications",
        "prescriptions": "/api/v1/prescriptions",
        "inventory": "/api/v1/inventory"
    }
}

# Model Configuration
MODEL_CONFIG = {
    'current_version': '1.0.0',
    'supported_versions': ['1.0.0', '1.1.0', '2.0.0'],
    'model_types': {
        'gnn': {
            'description': 'Graph Neural Network for drug-gene interaction prediction',
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score']
        },
        'ontology': {
            'description': 'Ontology-based model for drug classification',
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score']
        }
    },
    'version_control': {
        'path': 'models',
        'metadata_file': 'model_metadata.json'
    }
}

# Integration Configuration
INTEGRATION_CONFIG = {
    'emr': {
        'enabled': True,
        'api_url': 'https://emr-api.example.com',
        'api_key': '',  # To be set in environment variables
        'timeout': 30,
        'retry_attempts': 3
    },
    'lab': {
        'enabled': True,
        'api_url': 'https://lab-api.example.com',
        'api_key': '',  # To be set in environment variables
        'timeout': 30,
        'retry_attempts': 3
    },
    'pharmacy': {
        'enabled': True,
        'api_url': 'https://pharmacy-api.example.com',
        'api_key': '',  # To be set in environment variables
        'timeout': 30,
        'retry_attempts': 3
    }
}

# Security Configuration
SECURITY_CONFIG = {
    'https': {
        'enabled': True,
        'cert_path': 'certs/cert.pem',
        'key_path': 'certs/key.pem'
    },
    'authentication': {
        'enabled': True,
        'jwt_secret': '',  # To be set in environment variables
        'token_expiry': 3600  # 1 hour in seconds
    },
    'cors': {
        'enabled': True,
        'allowed_origins': ['https://example.com'],
        'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE'],
        'allowed_headers': ['Content-Type', 'Authorization']
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/app.log',
    'max_size': 10485760,  # 10MB
    'backup_count': 5
}

# Database Configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'genomedrugai',
    'user': '',  # To be set in environment variables
    'password': '',  # To be set in environment variables
    'pool_size': 5,
    'max_overflow': 10
}

# Cache Configuration
CACHE_CONFIG = {
    'enabled': True,
    'type': 'redis',
    'host': 'localhost',
    'port': 6379,
    'password': '',  # To be set in environment variables
    'ttl': 3600  # 1 hour in seconds
}

# API Rate Limiting
RATE_LIMIT_CONFIG = {
    'enabled': True,
    'requests_per_minute': 60,
    'burst_limit': 10
}

# Feature Flags
FEATURE_FLAGS = {
    'enable_advanced_analytics': True,
    'enable_batch_processing': True,
    'enable_real_time_updates': True,
    'enable_export_functionality': True
}

# Create necessary directories
for directory in [MODELS_DIR, CERT_DIR]:
    directory.mkdir(exist_ok=True) 