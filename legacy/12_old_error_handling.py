# Legacy: Old Error Handling Approaches (v0.1.0-rc10)
# Different error handling strategies tested during development
# Replaced by current structured error handling in v0.2.0

import sys
import traceback
import logging
from typing import Optional, Dict, Any, Union
from enum import Enum

# Approach 1: No Error Handling (Early Development)
class NoErrorHandlingChunker:
    """Chunker without error handling - crashes on any issue"""
    
    def chunk_text(self, text: str) -> list:
        """Chunk text without error handling"""
        # This would crash if text is None, not a string, etc.
        return text.split('. ')

# Problems: Crashes on any error, poor user experience, hard to debug

# Approach 2: Basic Try-Catch
class BasicErrorHandlingChunker:
    """Chunker with basic try-catch error handling"""
    
    def chunk_text(self, text: str) -> list:
        """Chunk text with basic error handling"""
        try:
            return text.split('. ')
        except Exception as e:
            print(f"Error chunking text: {e}")
            return []

# Problems: Generic error handling, no error types, poor error messages

# Approach 3: Custom Exception Classes
class ChunkingError(Exception):
    """Base exception for chunking errors"""
    pass

class InvalidInputError(ChunkingError):
    """Raised when input is invalid"""
    pass

class ProcessingError(ChunkingError):
    """Raised when processing fails"""
    pass

class CustomExceptionChunker:
    """Chunker with custom exception handling"""
    
    def chunk_text(self, text: str) -> list:
        """Chunk text with custom exceptions"""
        if text is None:
            raise InvalidInputError("Text cannot be None")
        
        if not isinstance(text, str):
            raise InvalidInputError(f"Text must be string, got {type(text)}")
        
        if not text.strip():
            raise InvalidInputError("Text cannot be empty")
        
        try:
            chunks = text.split('. ')
            if not chunks:
                raise ProcessingError("No chunks created")
            return chunks
        except Exception as e:
            raise ProcessingError(f"Failed to process text: {e}")

# Problems: Exceptions bubble up, user must handle them, verbose

# Approach 4: Result Pattern
class Result:
    """Result pattern for error handling"""
    
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
    
    @classmethod
    def success(cls, data: Any) -> 'Result':
        """Create successful result"""
        return cls(True, data=data)
    
    @classmethod
    def failure(cls, error: str) -> 'Result':
        """Create failure result"""
        return cls(False, error=error)
    
    def is_success(self) -> bool:
        """Check if result is successful"""
        return self.success
    
    def is_failure(self) -> bool:
        """Check if result is failure"""
        return not self.success
    
    def get_data(self) -> Any:
        """Get data if successful"""
        if not self.success:
            raise ValueError("Cannot get data from failed result")
        return self.data
    
    def get_error(self) -> str:
        """Get error if failed"""
        if self.success:
            raise ValueError("Cannot get error from successful result")
        return self.error

class ResultPatternChunker:
    """Chunker using result pattern for error handling"""
    
    def chunk_text(self, text: str) -> Result:
        """Chunk text returning Result object"""
        if text is None:
            return Result.failure("Text cannot be None")
        
        if not isinstance(text, str):
            return Result.failure(f"Text must be string, got {type(text)}")
        
        if not text.strip():
            return Result.failure("Text cannot be empty")
        
        try:
            chunks = text.split('. ')
            if not chunks:
                return Result.failure("No chunks created")
            return Result.success(chunks)
        except Exception as e:
            return Result.failure(f"Failed to process text: {e}")

# Problems: More verbose, requires checking result status, no automatic error propagation

# Approach 5: Logging-Based Error Handling
class LoggingErrorChunker:
    """Chunker with logging-based error handling"""
    
    def __init__(self):
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('chunker')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def chunk_text(self, text: str) -> list:
        """Chunk text with logging error handling"""
        try:
            if text is None:
                self.logger.error("Text cannot be None")
                return []
            
            if not isinstance(text, str):
                self.logger.error(f"Text must be string, got {type(text)}")
                return []
            
            if not text.strip():
                self.logger.warning("Text is empty, returning empty list")
                return []
            
            chunks = text.split('. ')
            self.logger.info(f"Successfully created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return []

# Problems: Silent failures, no error propagation, hard to handle programmatically

# Approach 6: Error Code Pattern
class ErrorCode(Enum):
    """Error codes for chunking operations"""
    SUCCESS = 0
    INVALID_INPUT = 1
    EMPTY_TEXT = 2
    PROCESSING_FAILED = 3
    UNKNOWN_ERROR = 4

class ErrorCodeChunker:
    """Chunker using error codes for error handling"""
    
    def chunk_text(self, text: str) -> tuple[list, ErrorCode, str]:
        """Chunk text returning (data, error_code, message)"""
        if text is None:
            return [], ErrorCode.INVALID_INPUT, "Text cannot be None"
        
        if not isinstance(text, str):
            return [], ErrorCode.INVALID_INPUT, f"Text must be string, got {type(text)}"
        
        if not text.strip():
            return [], ErrorCode.EMPTY_TEXT, "Text cannot be empty"
        
        try:
            chunks = text.split('. ')
            if not chunks:
                return [], ErrorCode.PROCESSING_FAILED, "No chunks created"
            return chunks, ErrorCode.SUCCESS, "Success"
        except Exception as e:
            return [], ErrorCode.UNKNOWN_ERROR, f"Unexpected error: {e}"

# Problems: Tuple returns, no type safety, error codes must be checked

# Approach 7: Context Manager Error Handling
class ErrorContext:
    """Context manager for error handling"""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.errors = []
    
    def __enter__(self):
        """Enter error context"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit error context"""
        if exc_type is not None:
            self.errors.append({
                'type': exc_type.__name__,
                'message': str(exc_val),
                'traceback': traceback.format_exc()
            })
            return False  # Don't suppress exception
        return True
    
    def add_error(self, error_type: str, message: str):
        """Add error to context"""
        self.errors.append({
            'type': error_type,
            'message': message,
            'traceback': None
        })
    
    def has_errors(self) -> bool:
        """Check if context has errors"""
        return len(self.errors) > 0
    
    def get_errors(self) -> list:
        """Get all errors"""
        return self.errors.copy()

class ContextErrorChunker:
    """Chunker using context manager for error handling"""
    
    def chunk_text(self, text: str) -> tuple[list, ErrorContext]:
        """Chunk text with error context"""
        context = ErrorContext("chunk_text")
        
        with context:
            if text is None:
                context.add_error("InvalidInput", "Text cannot be None")
                return [], context
            
            if not isinstance(text, str):
                context.add_error("InvalidInput", f"Text must be string, got {type(text)}")
                return [], context
            
            if not text.strip():
                context.add_error("EmptyInput", "Text cannot be empty")
                return [], context
            
            chunks = text.split('. ')
            if not chunks:
                context.add_error("ProcessingFailed", "No chunks created")
                return [], context
            
            return chunks, context

# Problems: Complex, verbose, no automatic error propagation

# Approach 8: Decorator-Based Error Handling
def handle_errors(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return None
    return wrapper

class DecoratorErrorChunker:
    """Chunker using decorator for error handling"""
    
    @handle_errors
    def chunk_text(self, text: str) -> list:
        """Chunk text with decorator error handling"""
        if text is None:
            raise ValueError("Text cannot be None")
        
        if not isinstance(text, str):
            raise ValueError(f"Text must be string, got {type(text)}")
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        chunks = text.split('. ')
        if not chunks:
            raise ValueError("No chunks created")
        
        return chunks

# Problems: Generic error handling, no error details, silent failures

# CURRENT APPROACH: Structured Error Handling
# ==========================================

"""
Current approach combines:
- Custom exceptions for specific error types
- Proper error messages and context
- Logging for debugging
- Graceful degradation where appropriate
- Clear error propagation

Example of current approach:
class Chunker:
    def chunk_text(self, text: str) -> List[str]:
        if text is None:
            raise ValueError("Text cannot be None")
        
        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got {type(text)}")
        
        if not text.strip():
            return []
        
        try:
            chunks = text.split('. ')
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            raise ChunkingError(f"Text chunking failed: {e}") from e
"""

# Migration Guide
# ===============

def migrate_from_no_handling():
    """Migrate from no error handling"""
    # OLD:
    # chunks = chunker.chunk_text(None)  # Crashes
    
    # NEW:
    # try:
    #     chunks = chunker.chunk_text(None)
    # except ValueError as e:
    #     print(f"Invalid input: {e}")
    pass

def migrate_from_basic_try_catch():
    """Migrate from basic try-catch"""
    # OLD:
    # try:
    #     chunks = chunker.chunk_text(text)
    # except Exception as e:
    #     print(f"Error: {e}")
    
    # NEW:
    # try:
    #     chunks = chunker.chunk_text(text)
    # except ValueError as e:
    #     print(f"Invalid input: {e}")
    # except ChunkingError as e:
    #     print(f"Chunking failed: {e}")
    pass

# Why Current Approach Was Chosen
# ===============================

REASONS_FOR_CURRENT_APPROACH = [
    "Clear error types and messages",
    "Proper exception hierarchy",
    "Automatic error propagation",
    "Good debugging support",
    "User-friendly error messages",
    "Graceful degradation where appropriate",
    "Follows Python best practices",
    "Easy to understand and maintain"
]
