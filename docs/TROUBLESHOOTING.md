# EchoGem Troubleshooting Guide

## 🚨 Common Issues and Solutions

This guide covers the most common problems users encounter with EchoGem and provides step-by-step solutions.

## 🔑 API Key Issues

### Problem: Google Gemini API Key Error
```
Error: Google API key required. Set GOOGLE_API_KEY environment variable.
```

**Solutions:**

1. **Set Environment Variable (Recommended)**
   ```bash
   # Windows (PowerShell)
   $env:GOOGLE_API_KEY="your_actual_api_key"
   
   # Windows (Command Prompt)
   set GOOGLE_API_KEY=your_actual_api_key
   
   # macOS/Linux
   export GOOGLE_API_KEY="your_actual_api_key"
   ```

2. **Verify API Key is Set**
   ```bash
   # Windows (PowerShell)
   echo $env:GOOGLE_API_KEY
   
   # Windows (Command Prompt)
   echo %GOOGLE_API_KEY%
   
   # macOS/Linux
   echo $GOOGLE_API_KEY
   ```

3. **Get New API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy and set the new key

4. **Check API Key Format**
   - API keys should start with `AIza...`
   - No spaces or special characters
   - Case-sensitive

### Problem: Pinecone API Key Error
```
Error: Pinecone API key required. Set PINECONE_API_KEY environment variable.
```

**Solutions:**

1. **Set Environment Variable**
   ```bash
   # Windows (PowerShell)
   $env:PINECONE_API_KEY="your_pinecone_api_key"
   
   # Windows (Command Prompt)
   set PINECONE_API_KEY=your_pinecone_api_key
   
   # macOS/Linux
   export PINECONE_API_KEY="your_pinecone_api_key"
   ```

2. **Get Pinecone API Key**
   - Visit [Pinecone Console](https://app.pinecone.io/)
   - Navigate to API Keys section
   - Copy your API key

3. **Verify Pinecone Account**
   - Ensure your Pinecone account is active
   - Check if you have available indexes

### Problem: Invalid API Key Format
```
Error: Invalid API key format
```

**Solutions:**

1. **Check Key Length**
   - Google Gemini: Usually 39 characters
   - Pinecone: Usually 64 characters

2. **Remove Extra Characters**
   - No quotes around the key
   - No trailing spaces
   - No line breaks

3. **Regenerate API Key**
   - Delete old key
   - Create new key
   - Update environment variable

## 🌐 Connection Issues

### Problem: Pinecone Connection Failed
```
Error: Failed to connect to Pinecone
Error: Connection timeout
```

**Solutions:**

1. **Check Internet Connection**
   ```bash
   # Test basic connectivity
   ping google.com
   
   # Test Pinecone connectivity
   curl -I https://app.pinecone.io/
   ```

2. **Verify Pinecone Service Status**
   - Check [Pinecone Status Page](https://status.pinecone.io/)
   - Look for service outages

3. **Check Firewall/Proxy Settings**
   - Ensure port 443 (HTTPS) is open
   - Configure proxy if behind corporate firewall

4. **Verify Region Settings**
   ```python
   from echogem import Processor
   
   # Try different regions
   processor = Processor(
       pinecone_api_key="your_key",
       region="us-west1-gcp"  # Alternative region
   )
   ```

### Problem: Google Gemini API Connection Issues
```
Error: Failed to connect to Google Gemini API
Error: API rate limit exceeded
```

**Solutions:**

1. **Check API Quotas**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Check your usage and limits

2. **Implement Rate Limiting**
   ```python
   import time
   
   # Add delays between API calls
   time.sleep(1)  # 1 second delay
   ```

3. **Use Retry Logic**
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
   def api_call_with_retry():
       # Your API call here
       pass
   ```

## 💾 Memory and Performance Issues

### Problem: Out of Memory Error
```
Error: Out of memory during processing
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce Chunk Size**
   ```python
   from echogem import ChunkingOptions
   
   options = ChunkingOptions(
       max_tokens=1000,  # Reduce from default 2000
       similarity_threshold=0.85
   )
   
   response = processor.process_transcript("transcript.txt", options)
   ```

2. **Process Smaller Files**
   - Split large transcripts into smaller files
   - Process files individually

3. **Increase System Memory**
   - Close other applications
   - Add more RAM if possible
   - Use virtual memory/swap

4. **Use Streaming Processing**
   ```python
   # Process in smaller batches
   def process_large_file(file_path, batch_size=1000):
       with open(file_path, 'r') as f:
           while True:
               chunk = f.read(batch_size)
               if not chunk:
                   break
               # Process chunk
               processor.process_transcript(chunk)
   ```

### Problem: Slow Processing
```
Processing is very slow
Long response times
```

**Solutions:**

1. **Optimize Chunking Parameters**
   ```python
   options = ChunkingOptions(
       max_tokens=1500,           # Smaller chunks
       similarity_threshold=0.80,  # Lower threshold
       coherence_threshold=0.70    # Lower coherence
   )
   ```

2. **Use Faster Embedding Models**
   ```python
   processor = Processor(
       embedding_model="all-MiniLM-L6-v2"  # Faster model
   )
   ```

3. **Enable Caching**
   ```python
   # Ensure usage cache is enabled
   processor = Processor(
       usage_cache_path="usage_cache.csv"
   )
   ```

4. **Batch Processing**
   ```python
   # Process multiple files in parallel
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=3) as executor:
       results = list(executor.map(processor.process_transcript, files))
   ```

## 📁 File and Path Issues

### Problem: File Not Found
```
Error: Transcript file not found
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**

1. **Check File Path**
   ```python
   import os
   
   # Verify file exists
   file_path = "transcript.txt"
   if os.path.exists(file_path):
       print(f"File exists: {file_path}")
   else:
       print(f"File not found: {file_path}")
   ```

2. **Use Absolute Paths**
   ```python
   import os
   
   # Get absolute path
   abs_path = os.path.abspath("transcript.txt")
   print(f"Absolute path: {abs_path}")
   ```

3. **Check File Permissions**
   ```python
   import os
   
   # Check if file is readable
   if os.access("transcript.txt", os.R_OK):
       print("File is readable")
   else:
       print("File is not readable")
   ```

4. **Handle Different File Formats**
   ```python
   # Support multiple file formats
   supported_formats = ['.txt', '.md', '.doc', '.docx']
   
   def get_file_extension(file_path):
       return os.path.splitext(file_path)[1].lower()
   
   if get_file_extension(file_path) not in supported_formats:
       print(f"Unsupported file format: {get_file_extension(file_path)}")
   ```

### Problem: Encoding Issues
```
Error: UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solutions:**

1. **Try Different Encodings**
   ```python
   encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
   
   for encoding in encodings:
       try:
           with open(file_path, 'r', encoding=encoding) as f:
               content = f.read()
           print(f"Successfully read with {encoding} encoding")
           break
       except UnicodeDecodeError:
           continue
   ```

2. **Detect Encoding**
   ```python
   import chardet
   
   with open(file_path, 'rb') as f:
       raw_data = f.read()
       detected = chardet.detect(raw_data)
       encoding = detected['encoding']
   
   with open(file_path, 'r', encoding=encoding) as f:
       content = f.read()
   ```

## 🐛 Code and Import Issues

### Problem: Import Errors
```
ImportError: cannot import name 'Processor' from 'echogem'
ModuleNotFoundError: No module named 'echogem'
```

**Solutions:**

1. **Check Installation**
   ```bash
   # Verify EchoGem is installed
   pip list | grep echogem
   
   # Reinstall if needed
   pip uninstall echogem
   pip install echogem
   ```

2. **Check Python Environment**
   ```bash
   # Verify Python path
   python -c "import sys; print(sys.path)"
   
   # Check if in virtual environment
   which python
   pip list
   ```

3. **Install from Source**
   ```bash
   # Clone and install from source
   git clone https://github.com/yourusername/echogem.git
   cd echogem
   pip install -e .
   ```

### Problem: Version Compatibility
```
AttributeError: 'Processor' object has no attribute 'new_method'
```

**Solutions:**

1. **Check Version**
   ```python
   import echogem
   print(f"EchoGem version: {echogem.__version__}")
   ```

2. **Update to Latest Version**
   ```bash
   pip install --upgrade echogem
   ```

3. **Check Documentation**
   - Review API changes in release notes
   - Check method signatures

## 🎨 Graph Visualization Issues

### Problem: Pygame Not Available
```
ImportError: No module named 'pygame'
```

**Solutions:**

1. **Install Pygame**
   ```bash
   pip install pygame
   ```

2. **Alternative Installation Methods**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install python3-pygame
   
   # On macOS with Homebrew
   brew install pygame
   
   # On Windows
   pip install pygame --pre
   ```

### Problem: Graph Window Not Displaying
```
Graph window appears but is blank
No nodes visible
```

**Solutions:**

1. **Check Data Availability**
   ```python
   # Ensure you have processed transcripts
   stats = processor.get_usage_statistics()
   print(f"Total chunks: {stats.get('total_chunks', 0)}")
   print(f"Total queries: {stats.get('total_queries', 0)}")
   ```

2. **Force Graph Refresh**
   ```python
   # Press 'R' key in graph window to reset view
   # Press 'F' key to toggle force-directed layout
   ```

3. **Check Display Settings**
   - Ensure display scaling is set correctly
   - Try different screen resolutions
   - Check if running in remote session

### Problem: Graph Performance Issues
```
Graph is slow or unresponsive
High CPU usage during visualization
```

**Solutions:**

1. **Reduce Graph Complexity**
   ```python
   # Limit number of nodes displayed
   # The graph automatically limits nodes for performance
   ```

2. **Use Different Layouts**
   - Press 'C' for circular layout (faster)
   - Press 'H' for hierarchical layout
   - Press 'F' for force-directed layout

3. **Close Other Applications**
   - Reduce system load
   - Free up memory and CPU

## 🔧 CLI Issues

### Problem: Command Not Found
```
'echogem' is not recognized as an internal or external command
```

**Solutions:**

1. **Use Python Module Syntax**
   ```bash
   python -m echogem.cli --help
   python -m echogem.cli process transcript.txt
   ```

2. **Check PATH**
   ```bash
   # Windows
   echo $env:PATH
   
   # macOS/Linux
   echo $PATH
   ```

3. **Reinstall with Entry Points**
   ```bash
   pip uninstall echogem
   pip install echogem
   ```

### Problem: CLI Arguments Not Working
```
Error: unrecognized arguments
```

**Solutions:**

1. **Check Help**
   ```bash
   python -m echogem.cli --help
   python -m echogem.cli process --help
   ```

2. **Use Correct Syntax**
   ```bash
   # Correct syntax
   python -m echogem.cli process transcript.txt --show-chunks
   
   # Not
   python -m echogem.cli --process transcript.txt
   ```

3. **Check Argument Order**
   ```bash
   # Arguments must come after subcommand
   python -m echogem.cli process transcript.txt --max-tokens 3000
   ```

## 📊 Data and Storage Issues

### Problem: Pinecone Index Issues
```
Error: Index not found
Error: Failed to create index
```

**Solutions:**

1. **Check Index Names**
   ```python
   # Use default index names
   processor = Processor(
       chunk_index_name="echogem-chunks",
       pa_index_name="echogem-pa"
   )
   ```

2. **Verify Pinecone Account**
   - Check account status
   - Verify available indexes
   - Check region settings

3. **Create Index Manually**
   ```python
   import pinecone
   
   pc = pinecone.Pinecone(api_key="your_key")
   
   # Create chunk index
   pc.create_index(
       name="echogem-chunks",
       dimension=384,
       spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
   )
   ```

### Problem: Usage Cache Issues
```
Error: Failed to write to usage cache
Cache file corrupted
```

**Solutions:**

1. **Check File Permissions**
   ```python
   import os
   
   cache_path = "usage_cache_store.csv"
   if os.access(cache_path, os.W_OK):
       print("Cache file is writable")
   else:
       print("Cache file is not writable")
   ```

2. **Clear Corrupted Cache**
   ```bash
   # Remove corrupted cache file
   rm usage_cache_store.csv
   
   # EchoGem will create new cache file
   ```

3. **Check Disk Space**
   ```bash
   # Check available disk space
   df -h
   ```

## 🚀 Performance Optimization

### Problem: High API Costs
```
API calls are expensive
High token usage
```

**Solutions:**

1. **Optimize Chunking**
   ```python
   options = ChunkingOptions(
       max_tokens=1500,           # Smaller chunks
       similarity_threshold=0.85,  # Higher quality
       coherence_threshold=0.80    # Better coherence
   )
   ```

2. **Use Caching Effectively**
   ```python
   # Process once, query many times
   response = processor.process_transcript("transcript.txt")
   
   # Multiple queries use cached chunks
   for question in questions:
       result = processor.query(question)
   ```

3. **Batch Similar Queries**
   ```python
   # Group similar questions
   similar_questions = [
       "What is the main topic?",
       "What are the key points?",
       "What are the conclusions?"
   ]
   
   for question in similar_questions:
       result = processor.query(question)
   ```

### Problem: Slow Vector Search
```
Vector similarity search is slow
Long retrieval times
```

**Solutions:**

1. **Optimize Search Parameters**
   ```python
   # Use fewer results for faster search
   chunks = processor.pick_chunks("query", k=3)  # Instead of k=10
   ```

2. **Use Efficient Embedding Models**
   ```python
   processor = Processor(
       embedding_model="all-MiniLM-L6-v2"  # Fast and efficient
   )
   ```

3. **Implement Search Caching**
   ```python
   # Cache search results
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def cached_search(query, k=5):
       return processor.pick_chunks(query, k)
   ```

## 🆘 Getting Additional Help

### Debug Mode

Enable debug logging for detailed error information:

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Process with verbose output
response = processor.process_transcript("transcript.txt")
```

### System Information

Collect system information for troubleshooting:

```python
import sys
import platform
import echogem

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"EchoGem version: {echogem.__version__}")
print(f"Python path: {sys.path}")
```

### Common Commands for Troubleshooting

```bash
# Check Python environment
python --version
pip list

# Check EchoGem installation
python -c "import echogem; print(echogem.__version__)"

# Test basic functionality
python -m echogem.cli --help

# Check environment variables
echo $GOOGLE_API_KEY
echo $PINECONE_API_KEY

# Verify file permissions
ls -la transcript.txt
```

### Reporting Issues

When reporting issues, include:

1. **Error Message**: Complete error traceback
2. **System Information**: OS, Python version, EchoGem version
3. **Steps to Reproduce**: Detailed steps to trigger the issue
4. **Expected vs Actual Behavior**: What you expected vs what happened
5. **Environment**: Virtual environment, dependencies, etc.

### Support Channels

- **GitHub Issues**: [Report bugs and request features](https://github.com/yourusername/echogem/issues)
- **GitHub Discussions**: [Ask questions and share solutions](https://github.com/yourusername/echogem/discussions)
- **Documentation**: Check the docs folder for detailed guides
- **Examples**: Review examples and demos for usage patterns

This troubleshooting guide covers the most common issues. If you encounter a problem not covered here, please report it through the support channels with detailed information about your environment and the specific error you're experiencing.
