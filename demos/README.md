# EchoGem Demos

This folder contains comprehensive demonstrations of the EchoGem library's capabilities across different use cases, text types, and scenarios.

## Demo Overview

### 🎯 **Core Functionality Demos**
- **`01_basic_workflow_demo.py`** - Complete end-to-end workflow demonstration
- **`02_cli_demo.py`** - Command-line interface usage examples
- **`03_api_demo.py`** - Python API usage examples

### 📚 **Text Type Demos**
- **`04_academic_paper_demo.py`** - Processing academic research papers
- **`05_meeting_transcript_demo.py`** - Business meeting transcript analysis
- **`06_podcast_demo.py`** - Long-form audio content processing
- **`07_technical_documentation_demo.py`** - Software documentation analysis
- **`08_news_articles_demo.py`** - News and article processing

### 🔍 **Advanced Feature Demos**
- **`09_performance_benchmarking_demo.py`** - Performance analysis and optimization
- **`10_custom_chunking_demo.py`** - Advanced chunking strategies
- **`11_retrieval_analysis_demo.py`** - Deep dive into retrieval mechanisms
- **`12_graph_visualization_demo.py`** - Interactive graph exploration
- **`13_batch_processing_demo.py`** - Large-scale document processing

### 📊 **Analysis & Statistics Demos**
- **`14_usage_analytics_demo.py`** - Usage pattern analysis
- **`15_quality_assessment_demo.py`** - Chunk and response quality metrics

## How to Use

1. **Install EchoGem**: `pip install -e .`
2. **Set up environment variables**:
   ```bash
   export GOOGLE_API_KEY="your_key_here"
   export PINECONE_API_KEY="your_key_here"
   ```
3. **Run any demo**: `python demos/01_basic_workflow_demo.py`

## Demo Data

Each demo includes sample data or instructions for obtaining test data. Some demos create their own sample content for demonstration purposes.

## Performance Metrics

Several demos include performance benchmarking:
- Processing time per document
- Memory usage
- Chunk quality scores
- Retrieval accuracy
- Response generation time

## Customization

All demos are designed to be easily customizable:
- Modify text sources
- Adjust chunking parameters
- Change retrieval strategies
- Customize visualization options

## Troubleshooting

If you encounter issues:
1. Check your API keys are set correctly
2. Ensure all dependencies are installed
3. Verify Pinecone index configuration
4. Check the demo-specific error handling sections
