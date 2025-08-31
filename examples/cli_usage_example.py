#!/usr/bin/env python3
"""
EchoGem Library Usage Example
=============================

This example demonstrates how to use the EchoGem library to:
1. Process a transcript file
2. Ask questions and get answers
3. Use the CLI interface
4. Visualize the information graph
5. Work with the library programmatically

Prerequisites:
- Set GOOGLE_API_KEY environment variable
- Set PINECONE_API_KEY environment variable
- Have a transcript.txt file ready
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path to import echogem
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def setup_environment():
    """Check if required environment variables are set"""
    required_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables:")
        print("   export GOOGLE_API_KEY='your-google-api-key'")
        print("   export PINECONE_API_KEY='your-pinecone-api-key'")
        return False
    
    print("✅ Environment variables configured")
    return True

def demonstrate_cli_usage():
    """Show how to use the CLI commands"""
    print("\n" + "="*60)
    print("CLI USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Process a transcript file:")
    print("   py -m echogem.cli process transcript.txt")
    print("   py -m echogem.cli process transcript.txt --show-chunks")
    
    print("\n2. Ask a question:")
    print("   py -m echogem.cli ask 'What is the main topic discussed?'")
    print("   py -m echogem.cli ask 'Who are the key speakers?' --show-chunks --show-metadata")
    
    print("\n3. Interactive mode:")
    print("   py -m echogem.cli interactive")
    
    print("\n4. Visualize information graph:")
    print("   py -m echogem.cli graph")
    print("   py -m echogem.cli graph --width 1400 --height 900")
    print("   py -m echogem.cli graph --export graph_data.json")
    
    print("\n5. System information:")
    print("   py -m echogem.cli stats")
    print("   py -m echogem.cli clear")

def demonstrate_programmatic_usage():
    """Show how to use the library programmatically"""
    print("\n" + "="*60)
    print("PROGRAMMATIC USAGE EXAMPLES")
    print("="*60)
    
    try:
        from echogem.processor import Processor
        from echogem.models import ChunkingOptions, QueryOptions
        
        print("\n1. Initialize the processor:")
        print("   processor = Processor()")
        
        print("\n2. Process a transcript:")
        print("   processor.chunk_and_process('transcript.txt', output_chunks=True)")
        
        print("\n3. Ask questions:")
        print("   result = processor.answer_question('What is the main topic?')")
        print("   print(result.answer)")
        
        print("\n4. Get system statistics:")
        print("   stats = processor.get_stats()")
        print("   print(stats)")
        
        print("\n5. Custom chunking options:")
        print("   options = ChunkingOptions(")
        print("       chunk_size=1000,")
        print("       overlap=200,")
        print("       semantic_chunking=True")
        print("   )")
        
        print("\n6. Custom query options:")
        print("   query_opts = QueryOptions(")
        print("       top_k=5,")
        print("       similarity_threshold=0.7,")
        print("       include_metadata=True")
        print("   )")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the correct directory")

def demonstrate_workflow():
    """Show a complete workflow example"""
    print("\n" + "="*60)
    print("COMPLETE WORKFLOW EXAMPLE")
    print("="*60)
    
    print("\nStep 1: Set up environment variables")
    print("   export GOOGLE_API_KEY='your-google-api-key'")
    print("   export PINECONE_API_KEY='your-pinecone-api-key'")
    
    print("\nStep 2: Process your transcript")
    print("   py -m echogem.cli process transcript.txt --show-chunks")
    
    print("\nStep 3: Ask questions interactively")
    print("   py -m echogem.cli interactive")
    
    print("\nStep 4: Visualize the information flow")
    print("   py -m echogem.cli graph")
    
    print("\nStep 5: Get detailed answers with metadata")
    print("   py -m echogem.cli ask 'What are the key insights?' --show-chunks --show-metadata")

def main():
    """Main function to run all demonstrations"""
    print("🚀 EchoGem Library Usage Examples")
    print("="*60)
    
    # Check environment
    if not setup_environment():
        print("\n⚠️  Please set the required environment variables first.")
        print("   You can still view the usage examples below.")
    
    # Show all usage examples
    demonstrate_cli_usage()
    demonstrate_programmatic_usage()
    demonstrate_workflow()
    
    print("\n" + "="*60)
    print("QUICK START COMMANDS")
    print("="*60)
    print("\n1. Process transcript: py -m echogem.cli process transcript.txt")
    print("2. Ask question: py -m echogem.cli ask 'Your question here?'")
    print("3. Interactive mode: py -m echogem.cli interactive")
    print("4. Visualize: py -m echogem.cli graph")
    
    print("\n📚 For more details, check the documentation:")
    print("   - User Guide: echogem/docs/USER_GUIDE.md")
    print("   - API Reference: echogem/docs/API_REFERENCE.md")
    print("   - Architecture: echogem/docs/ARCHITECTURE.md")

if __name__ == "__main__":
    main()
