# Legacy: Old Configuration Approaches (v0.1.0-rc11)
# Different configuration management strategies tested
# Replaced by current environment variable and config file approach in v0.2.0

import os
import json
import yaml
import configparser
from typing import Dict, Any, Optional
from pathlib import Path

# Approach 1: Hardcoded Configuration
class HardcodedConfig:
    """Configuration with hardcoded values - no flexibility"""
    
    # These values were hardcoded throughout the code
    CHUNK_SIZE = 1000
    OVERLAP = 100
    MAX_CHUNKS = 1000
    VECTOR_DIMENSION = 768
    SIMILARITY_THRESHOLD = 0.7
    TOP_K_RESULTS = 5
    
    # API keys were hardcoded (security risk!)
    GOOGLE_API_KEY = "your-api-key-here"
    PINECONE_API_KEY = "your-pinecone-key-here"
    PINECONE_ENVIRONMENT = "us-west1-gcp"

# Problems: No flexibility, security risks, hard to change, no environment support

# Approach 2: Environment Variables Only
class EnvOnlyConfig:
    """Configuration using only environment variables"""
    
    def __init__(self):
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.overlap = int(os.getenv('OVERLAP', '100'))
        self.max_chunks = int(os.getenv('MAX_CHUNKS', '1000'))
        self.vector_dimension = int(os.getenv('VECTOR_DIMENSION', '768'))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
        self.top_k_results = int(os.getenv('TOP_K_RESULTS', '5'))
        
        # API keys
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
    
    def validate(self):
        """Validate configuration values"""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.overlap < 0:
            raise ValueError("OVERLAP must be non-negative")
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")

# Problems: No defaults, hard to manage many variables, no hierarchical structure

# Approach 3: JSON Configuration
class JSONConfig:
    """Configuration using JSON files"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_file):
            # Create default config
            default_config = {
                "chunking": {
                    "chunk_size": 1000,
                    "overlap": 100,
                    "max_chunks": 1000
                },
                "vector_store": {
                    "dimension": 768,
                    "similarity_threshold": 0.7,
                    "top_k_results": 5
                },
                "api": {
                    "google_api_key": "",
                    "pinecone_api_key": "",
                    "pinecone_environment": "us-west1-gcp"
                }
            }
            self._save_config(default_config)
            return default_config
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        self._save_config(self.config)

# Problems: No type validation, no environment variable override, manual file management

# Approach 4: YAML Configuration
class YAMLConfig:
    """Configuration using YAML files"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_file):
            # Create default config
            default_config = {
                "chunking": {
                    "chunk_size": 1000,
                    "overlap": 100,
                    "max_chunks": 1000
                },
                "vector_store": {
                    "dimension": 768,
                    "similarity_threshold": 0.7,
                    "top_k_results": 5
                },
                "api": {
                    "google_api_key": "${GOOGLE_API_KEY}",
                    "pinecone_api_key": "${PINECONE_API_KEY}",
                    "pinecone_environment": "us-west1-gcp"
                }
            }
            self._save_config(default_config)
            return default_config
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                return self._resolve_environment_vars(config)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _resolve_environment_vars(self, config: Any) -> Any:
        """Resolve environment variable references in config"""
        if isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, '')
        elif isinstance(config, dict):
            return {k: self._resolve_environment_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_environment_vars(item) for item in config]
        else:
            return config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to YAML file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

# Problems: Additional dependency (PyYAML), no type validation, complex environment resolution

# Approach 5: INI Configuration
class INIConfig:
    """Configuration using INI files"""
    
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from INI file"""
        if not os.path.exists(self.config_file):
            # Create default config
            self._create_default_config()
        
        self.config.read(self.config_file)
    
    def _create_default_config(self):
        """Create default INI configuration"""
        self.config['chunking'] = {
            'chunk_size': '1000',
            'overlap': '100',
            'max_chunks': '1000'
        }
        
        self.config['vector_store'] = {
            'dimension': '768',
            'similarity_threshold': '0.7',
            'top_k_results': '5'
        }
        
        self.config['api'] = {
            'google_api_key': '',
            'pinecone_api_key': '',
            'pinecone_environment': 'us-west1-gcp'
        }
        
        self._save_config()
    
    def _save_config(self):
        """Save configuration to INI file"""
        try:
            with open(self.config_file, 'w') as f:
                self.config.write(f)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, section: str, key: str, fallback: Any = None) -> str:
        """Get configuration value"""
        return self.config.get(section, key, fallback=fallback)
    
    def getint(self, section: str, key: str, fallback: int = None) -> int:
        """Get integer configuration value"""
        return self.config.getint(section, key, fallback=fallback)
    
    def getfloat(self, section: str, key: str, fallback: float = None) -> float:
        """Get float configuration value"""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def set(self, section: str, key: str, value: str):
        """Set configuration value"""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))
        self._save_config()

# Problems: No nested structure, limited data types, no environment variable support

# Approach 6: Class-Based Configuration
class ClassBasedConfig:
    """Configuration using Python classes"""
    
    class ChunkingConfig:
        chunk_size: int = 1000
        overlap: int = 100
        max_chunks: int = 1000
    
    class VectorStoreConfig:
        dimension: int = 768
        similarity_threshold: float = 0.7
        top_k_results: int = 5
    
    class APIConfig:
        google_api_key: str = ""
        pinecone_api_key: str = ""
        pinecone_environment: str = "us-west1-gcp"
    
    def __init__(self):
        self.chunking = self.ChunkingConfig()
        self.vector_store = self.VectorStoreConfig()
        self.api = self.APIConfig()
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Chunking
        if os.getenv('CHUNK_SIZE'):
            self.chunking.chunk_size = int(os.getenv('CHUNK_SIZE'))
        if os.getenv('OVERLAP'):
            self.chunking.overlap = int(os.getenv('OVERLAP'))
        if os.getenv('MAX_CHUNKS'):
            self.chunking.max_chunks = int(os.getenv('MAX_CHUNKS'))
        
        # Vector store
        if os.getenv('VECTOR_DIMENSION'):
            self.vector_store.dimension = int(os.getenv('VECTOR_DIMENSION'))
        if os.getenv('SIMILARITY_THRESHOLD'):
            self.vector_store.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD'))
        if os.getenv('TOP_K_RESULTS'):
            self.vector_store.top_k_results = int(os.getenv('TOP_K_RESULTS'))
        
        # API
        if os.getenv('GOOGLE_API_KEY'):
            self.api.google_api_key = os.getenv('GOOGLE_API_KEY')
        if os.getenv('PINECONE_API_KEY'):
            self.api.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if os.getenv('PINECONE_ENVIRONMENT'):
            self.api.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')

# Problems: No persistence, no validation, hard to serialize

# Approach 7: Hybrid Configuration
class HybridConfig:
    """Hybrid configuration approach"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration with fallbacks"""
        # 1. Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        # 2. Override with environment variables
        self._override_from_environment()
        
        # 3. Set defaults for missing values
        self._set_defaults()
    
    def _override_from_environment(self):
        """Override config with environment variables"""
        env_mappings = {
            'CHUNK_SIZE': ('chunking', 'chunk_size'),
            'OVERLAP': ('chunking', 'overlap'),
            'MAX_CHUNKS': ('chunking', 'max_chunks'),
            'VECTOR_DIMENSION': ('vector_store', 'dimension'),
            'SIMILARITY_THRESHOLD': ('vector_store', 'similarity_threshold'),
            'TOP_K_RESULTS': ('vector_store', 'top_k_results'),
            'GOOGLE_API_KEY': ('api', 'google_api_key'),
            'PINECONE_API_KEY': ('api', 'pinecone_api_key'),
            'PINECONE_ENVIRONMENT': ('api', 'pinecone_environment')
        }
        
        for env_var, (section, key) in env_mappings.items():
            if os.getenv(env_var):
                if section not in self.config:
                    self.config[section] = {}
                self.config[section][key] = os.getenv(env_var)
    
    def _set_defaults(self):
        """Set default values for missing configuration"""
        defaults = {
            'chunking': {
                'chunk_size': 1000,
                'overlap': 100,
                'max_chunks': 1000
            },
            'vector_store': {
                'dimension': 768,
                'similarity_threshold': 0.7,
                'top_k_results': 5
            },
            'api': {
                'google_api_key': '',
                'pinecone_api_key': '',
                'pinecone_environment': 'us-west1-gcp'
            }
        }
        
        for section, values in defaults.items():
            if section not in self.config:
                self.config[section] = {}
            for key, value in values.items():
                if key not in self.config[section]:
                    self.config[section][key] = value

# Problems: Complex logic, no validation, hard to maintain

# CURRENT APPROACH: Environment Variables + Simple Config
# =====================================================

"""
Current approach combines:
- Environment variables for sensitive data (API keys)
- Simple configuration files for non-sensitive settings
- Clear precedence: env vars > config file > defaults
- Type validation and error handling
- Easy to use and understand

Example of current approach:
import os
from typing import Optional

class Config:
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
"""

# Migration Guide
# ===============

def migrate_from_hardcoded():
    """Migrate from hardcoded configuration"""
    # OLD:
    # CHUNK_SIZE = 1000  # Hardcoded in code
    
    # NEW:
    # chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
    pass

def migrate_from_json_config():
    """Migrate from JSON configuration"""
    # OLD:
    # config = JSONConfig("config.json")
    # chunk_size = config.get("chunking.chunk_size")
    
    # NEW:
    # chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
    pass

# Why Current Approach Was Chosen
# ===============================

REASONS_FOR_CURRENT_APPROACH = [
    "Environment variables for sensitive data (security best practice)",
    "Simple and standard approach",
    "Easy to deploy in different environments",
    "No additional dependencies",
    "Clear precedence rules",
    "Easy to understand and maintain",
    "Follows 12-factor app principles",
    "Works well with containerization and cloud deployment"
]
