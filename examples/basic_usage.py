#!/usr/bin/env python3
"""
Basic usage example for EchoGem
"""

import os
from echogem import Processor

def main():
    """Demonstrate basic EchoGem functionality"""
    
    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Please set PINECONE_API_KEY environment variable")
        return
    
    print("🚀 Initializing EchoGem...")
    
    try:
        # Initialize processor
        processor = Processor()
        print("✅ Processor initialized successfully")
        
        # Check if we have a transcript file
        transcript_file = "transcript.txt"
        if not os.path.exists(transcript_file):
            print(f"❌ Transcript file '{transcript_file}' not found")
            print("Please create a transcript.txt file or modify the script to use your file")
            return
        
        # Process transcript (only do this once)
        print(f"\n📝 Processing transcript: {transcript_file}")
        processor.chunk_and_process(transcript_file, output_chunks=True)
        
        # Ask some questions
        questions = [
            "What is the main topic discussed?",
            "What are the key points mentioned?",
            "Who are the main speakers?",
            "What conclusions were reached?"
        ]
        
        print(f"\n🤖 Asking {len(questions)} questions...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i} ---")
            print(f"Q: {question}")
            
            result = processor.answer_question(
                question,
                show_chunks=False,  # Set to True to see retrieved chunks
                show_metadata=False  # Set to True to see chunk metadata
            )
            
            print(f"A: {result.answer}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Chunks used: {len(result.chunks_used)}")
        
        # Show system statistics
        print(f"\n📊 System Statistics:")
        stats = processor.get_stats()
        print(f"  Usage cache size: {stats.get('usage_cache_size', 'N/A')}")
        
        print("\n✅ Example completed successfully!")
        print("\n💡 Try running 'echogem interactive' for an interactive questioning session")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure your API keys are set correctly")
        print("2. Check that you have a transcript.txt file")
        print("3. Ensure you have an internet connection")
        print("4. Verify your Pinecone account is active")

if __name__ == "__main__":
    main()
