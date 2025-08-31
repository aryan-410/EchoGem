# Legacy: Old Data Models (v0.1.0-rc8)
# Different approaches to data modeling and validation
# Replaced by Pydantic-based models in v0.2.0

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
import json
from datetime import datetime
import uuid

# Approach 1: Simple Dataclasses (no validation)
@dataclass
class SimpleChunk:
    """Simple chunk without validation"""
    id: str
    content: str
    title: str
    timestamp: str
    usage_count: int = 0

# Problems: No validation, no type checking, no serialization

# Approach 2: Manual Validation
class ManualChunk:
    """Chunk with manual validation"""
    
    def __init__(self, id: str, content: str, title: str, timestamp: str, usage_count: int = 0):
        # Manual validation
        if not id or not isinstance(id, str):
            raise ValueError("ID must be a non-empty string")
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
        if not title or not isinstance(title, str):
            raise ValueError("Title must be a non-empty string")
        if not isinstance(usage_count, int) or usage_count < 0:
            raise ValueError("Usage count must be a non-negative integer")
        
        self.id = id
        self.content = content
        self.title = title
        self.timestamp = timestamp
        self.usage_count = usage_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'content': self.content,
            'title': self.title,
            'timestamp': self.timestamp,
            'usage_count': self.usage_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManualChunk':
        """Create from dictionary"""
        return cls(**data)

# Problems: Verbose, repetitive validation code, no automatic serialization

# Approach 3: Attrs Library
try:
    import attr
    
    @attr.s
    class AttrsChunk:
        """Chunk using attrs library"""
        id: str = attr.ib(validator=attr.validators.instance_of(str))
        content: str = attr.ib(validator=attr.validators.instance_of(str))
        title: str = attr.ib(validator=attr.validators.instance_of(str))
        timestamp: str = attr.ib(validator=attr.validators.instance_of(str))
        usage_count: int = attr.ib(default=0, validator=attr.validators.ge(0))
        
        def to_dict(self) -> Dict[str, Any]:
            return attr.asdict(self)
        
        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'AttrsChunk':
            return cls(**data)

except ImportError:
    # Fallback if attrs not available
    class AttrsChunk:
        """Fallback attrs chunk"""
        pass

# Problems: Additional dependency, less intuitive than dataclasses

# Approach 4: Custom Base Class
class ValidatedModel:
    """Base class for validated models"""
    
    def __init__(self, **kwargs):
        self._validate_and_set(kwargs)
    
    def _validate_and_set(self, data: Dict[str, Any]):
        """Validate and set attributes"""
        for field_name, field_type in self.__annotations__.items():
            if field_name in data:
                value = data[field_name]
                if not isinstance(value, field_type):
                    raise ValueError(f"{field_name} must be {field_type}, got {type(value)}")
                setattr(self, field_name, value)
            elif hasattr(self, field_name):
                # Keep default value
                pass
            else:
                raise ValueError(f"Missing required field: {field_name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        for field_name in self.__annotations__:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if hasattr(value, 'to_dict'):
                    result[field_name] = value.to_dict()
                else:
                    result[field_name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidatedModel':
        """Create from dictionary"""
        return cls(**data)

class ValidatedChunk(ValidatedModel):
    """Chunk using custom validation base class"""
    id: str
    content: str
    title: str
    timestamp: str
    usage_count: int = 0

# Problems: Complex base class, no automatic serialization, manual field definitions

# Approach 5: JSON Schema Validation
class JSONSchemaChunk:
    """Chunk with JSON schema validation"""
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "content": {"type": "string", "minLength": 1},
            "title": {"type": "string", "minLength": 1},
            "timestamp": {"type": "string"},
            "usage_count": {"type": "integer", "minimum": 0}
        },
        "required": ["id", "content", "title", "timestamp"]
    }
    
    def __init__(self, **kwargs):
        self._validate_schema(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _validate_schema(self, data: Dict[str, Any]):
        """Validate against JSON schema"""
        try:
            import jsonschema
            jsonschema.validate(data, self.SCHEMA)
        except ImportError:
            # Fallback validation
            self._fallback_validation(data)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Validation error: {e}")
    
    def _fallback_validation(self, data: Dict[str, Any]):
        """Simple fallback validation"""
        required_fields = ['id', 'content', 'title', 'timestamp']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(data[field], str):
                raise ValueError(f"{field} must be a string")
        
        if 'usage_count' in data and not isinstance(data['usage_count'], int):
            raise ValueError("usage_count must be an integer")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {attr: getattr(self, attr) for attr in self.SCHEMA['properties']}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JSONSchemaChunk':
        """Create from dictionary"""
        return cls(**data)

# Problems: Complex schema definition, additional dependency, verbose

# Approach 6: Type Annotations with Runtime Checking
from typing import get_type_hints, get_origin, get_args

class TypeCheckedModel:
    """Base class with runtime type checking"""
    
    def __init__(self, **kwargs):
        type_hints = get_type_hints(self.__class__)
        for field_name, field_type in type_hints.items():
            if field_name in kwargs:
                value = kwargs[field_name]
                self._validate_type(field_name, value, field_type)
                setattr(self, field_name, value)
            elif hasattr(self, field_name):
                # Keep default value
                pass
            else:
                raise ValueError(f"Missing required field: {field_name}")
    
    def _validate_type(self, field_name: str, value: Any, expected_type: type):
        """Validate field type"""
        if get_origin(expected_type) is Union:
            # Handle Optional types
            if type(None) in get_args(expected_type):
                if value is None:
                    return
                # Check other types
                other_types = [t for t in get_args(expected_type) if t is not type(None)]
                if not any(isinstance(value, t) for t in other_types):
                    raise ValueError(f"{field_name} must be one of {other_types}, got {type(value)}")
            else:
                if not any(isinstance(value, t) for t in get_args(expected_type)):
                    raise ValueError(f"{field_name} must be one of {get_args(expected_type)}, got {type(value)}")
        else:
            if not isinstance(value, expected_type):
                raise ValueError(f"{field_name} must be {expected_type}, got {type(value)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        for field_name in get_type_hints(self.__class__):
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if hasattr(value, 'to_dict'):
                    result[field_name] = value.to_dict()
                else:
                    result[field_name] = value
        return result

class TypeCheckedChunk(TypeCheckedModel):
    """Chunk with runtime type checking"""
    id: str
    content: str
    title: str
    timestamp: str
    usage_count: int = 0

# Problems: Complex type checking logic, no automatic serialization

# Approach 7: Simple Dictionary with Validation
class DictChunk:
    """Chunk as validated dictionary"""
    
    def __init__(self, **kwargs):
        self._data = self._validate_and_clean(kwargs)
    
    def _validate_and_clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean input data"""
        # Required fields
        required = ['id', 'content', 'title', 'timestamp']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(data[field], str):
                raise ValueError(f"{field} must be a string")
            if not data[field].strip():
                raise ValueError(f"{field} cannot be empty")
        
        # Optional fields with defaults
        result = {
            'id': data['id'].strip(),
            'content': data['content'].strip(),
            'title': data['title'].strip(),
            'timestamp': data['timestamp'].strip(),
            'usage_count': data.get('usage_count', 0)
        }
        
        # Validate usage_count
        if not isinstance(result['usage_count'], int) or result['usage_count'] < 0:
            raise ValueError("usage_count must be a non-negative integer")
        
        return result
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to data"""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self._data.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DictChunk':
        """Create from dictionary"""
        return cls(**data)

# Problems: No type hints, attribute access through __getattr__, less intuitive

# CURRENT APPROACH: Pydantic Models
# =================================

"""
Pydantic provides:
- Automatic validation
- Type hints
- JSON serialization/deserialization
- IDE support
- Clear error messages
- Easy to use and understand

Example of current approach:
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Chunk(BaseModel):
    id: str
    content: str
    title: str
    timestamp_range: str
    usage_count: int = 0
    last_used: Optional[datetime] = None
"""

# Migration Guide
# ===============

def migrate_from_simple_dataclass():
    """Migrate from simple dataclass"""
    # OLD:
    # chunk = SimpleChunk("123", "content", "title", "00:00-01:00")
    
    # NEW:
    # chunk = Chunk(id="123", content="content", title="title", timestamp_range="00:00-01:00")
    pass

def migrate_from_manual_validation():
    """Migrate from manual validation"""
    # OLD:
    # chunk = ManualChunk("123", "content", "title", "00:00-01:00")
    
    # NEW:
    # chunk = Chunk(id="123", content="content", title="title", timestamp_range="00:00-01:00")
    pass

# Why Pydantic Was Chosen
# ========================

REASONS_FOR_PYDANTIC = [
    "Automatic validation with clear error messages",
    "Built-in JSON serialization/deserialization",
    "Excellent IDE support with type hints",
    "Active development and community support",
    "Performance optimized",
    "Easy to extend and customize",
    "Well documented and tested"
]
