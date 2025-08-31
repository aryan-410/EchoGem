#!/usr/bin/env python3
"""
Advanced usage example for EchoGem
"""

import os
import numpy as np
from echogem import Processor, ChunkingOptions, QueryOptions

def main():
    """Demonstrate advanced EchoGem functionality"""
    
    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Please set PINECONE_API_KEY environment variable")
        return
    
    print("🚀 Initializing EchoGem with advanced configuration...")
    
    try:
        # Initialize processor with custom configuration
        processor = Processor(
            chunk_index_name="advanced-chunks",
            pa_index_name="advanced-qa",
            usage_cache_path="advanced_usage_cache.csv"
        )
        print("✅ Advanced processor initialized successfully")
        
        # Check if we have a transcript file
        transcript_file = "transcript.txt"
        if not os.path.exists(transcript_file):
            print(f"❌ Transcript file '{transcript_file}' not found")
            print("Please create a transcript.txt file or modify the script to use your file")
            return
        
        # Process transcript with custom chunking
        print(f"\n📝 Processing transcript: {transcript_file}")
        processor.chunk_and_process(transcript_file, output_chunks=True)
        
        # Demonstrate different chunk retrieval strategies
        print(f"\n🔍 Testing different retrieval strategies...")
        
        strategies = [
            ("Balanced", {"entropy_weight": 0.25, "recency_weight": 0.25}),
            ("Content-focused", {"entropy_weight": 0.5, "recency_weight": 0.1}),
            ("Recency-focused", {"entropy_weight": 0.1, "recency_weight": 0.5}),
            ("Similarity-focused", {"entropy_weight": 0.1, "recency_weight": 0.1})
        ]
        
        test_question = "What are the main topics discussed?"
        
        for strategy_name, weights in strategies:
            print(f"\n--- {strategy_name} Strategy ---")
            print(f"Weights: {weights}")
            
            chunks = processor.pick_chunks(
                test_question,
                k=3,
                **weights
            )
            
            if chunks:
                print(f"Retrieved {len(chunks)} chunks:")
                for i, chunk in enumerate(chunks, 1):
                    print(f"  {i}. {chunk.title}")
                    print(f"     Keywords: {chunk.keywords}")
                    print(f"     Entities: {chunk.named_entities}")
            else:
                print("No chunks found")
        
        # Test different numbers of chunks
        print(f"\n📊 Testing different chunk counts...")
        
        for k in [1, 3, 5, 10]:
            print(f"\n--- Retrieving {k} chunks ---")
            chunks = processor.pick_chunks(test_question, k=k)
            if chunks:
                print(f"Retrieved {len(chunks)} chunks")
                # Show first chunk title as example
                if chunks:
                    print(f"First chunk: {chunks[0].title}")
            else:
                print("No chunks found")
        
        # Demonstrate Q&A pair retrieval
        print(f"\n💬 Testing Q&A pair retrieval...")
        
        # First, ask a few questions to build up the Q&A database
        questions = [
            "What is the main topic?",
            "Who are the speakers?",
            "What are the key conclusions?"
        ]
        
        for question in questions:
            result = processor.answer_question(question)
            print(f"Q: {question}")
            print(f"A: {result.answer[:100]}...")
        
        # Now test retrieving similar Q&A pairs
        print(f"\n🔍 Finding similar Q&A pairs...")
        
        test_query = "What was discussed about the main topic?"
        similar_qa = processor.get_similar_qa_pairs(
            test_query,
            k=3,
            sim_weight=0.7,
            entropy_weight=0.2,
            recency_weight=0.1
        )
        
        if similar_qa:
            print(f"Found {len(similar_qa)} similar Q&A pairs:")
            for i, (pair, score, parts) in enumerate(similar_qa, 1):
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   Q: {pair.prompt[:80]}...")
                print(f"   A: {pair.answer[:80]}...")
                print(f"   Breakdown: sim={parts['sim']:.2f}, ent={parts['ent']:.2f}, rec={parts['rec']:.2f}")
        else:
            print("No similar Q&A pairs found")
        
        # Show detailed statistics
        print(f"\n📊 Detailed System Statistics:")
        stats = processor.get_stats()
        print(f"  Usage cache size: {stats.get('usage_cache_size', 'N/A')}")
        print(f"  Chunk index: {stats.get('chunks', 'N/A')}")
        print(f"  PA index: {stats.get('prompt_answers', 'N/A')}")
        
        print("\n✅ Advanced example completed successfully!")
        print("\n💡 Try these advanced features:")
        print("  - Customize chunking parameters in the Chunker class")
        print("  - Adjust scoring weights for different use cases")
        print("  - Use different Pinecone indexes for different projects")
        print("  - Implement custom embedding models")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure your API keys are set correctly")
        print("2. Check that you have a transcript.txt file")
        print("3. Ensure you have an internet connection")
        print("4. Verify your Pinecone account is active")
        print("5. Check that you have sufficient Pinecone quota")

if __name__ == "__main__":
    main()
