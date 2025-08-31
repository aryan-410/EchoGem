#!/usr/bin/env python3
"""
Advanced Chunk Management Demo

This demo showcases EchoGem's advanced chunk management capabilities:
- merge_chunks: Combines overly similar chunks to reduce fragmentation
- chunk_radius: Measures how clustered a chunk is with its neighbors
- should_merge_chunks: Determines if chunks should be merged based on relevance and coherence

Prerequisites:
- Google AI API Key (free tier available)
- Pinecone API Key
- Sample transcript file

Usage:
    python 08_advanced_chunk_management_demo.py
"""

import os
import sys
from datetime import datetime

# Add the parent directory to the path to import echogem
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from echogem import Processor, Chunk, ChunkingOptions

def create_sample_chunks():
    """Create sample chunks for demonstration"""
    chunks = [
        Chunk(
            chunk_id="chunk_001",
            title="Introduction to AI",
            content="Artificial intelligence is transforming the way we work and live. Machine learning algorithms are becoming more sophisticated every day.",
            keywords=["artificial intelligence", "machine learning", "algorithms"],
            named_entities=["AI", "ML"],
            timestamp_range="00:00-00:30"
        ),
        Chunk(
            chunk_id="chunk_002",
            title="AI Applications",
            content="AI applications are everywhere in modern technology. From recommendation systems to autonomous vehicles, AI is reshaping industries.",
            keywords=["applications", "technology", "recommendation systems"],
            named_entities=["AI", "autonomous vehicles"],
            timestamp_range="00:30-01:00"
        ),
        Chunk(
            chunk_id="chunk_003",
            title="Machine Learning Basics",
            content="Machine learning is a subset of artificial intelligence. It involves training algorithms on data to make predictions.",
            keywords=["machine learning", "algorithms", "predictions"],
            named_entities=["ML", "AI"],
            timestamp_range="01:00-01:30"
        ),
        Chunk(
            chunk_id="chunk_004",
            title="Deep Learning",
            content="Deep learning uses neural networks with multiple layers. These networks can learn complex patterns in data.",
            keywords=["deep learning", "neural networks", "patterns"],
            named_entities=["neural networks"],
            timestamp_range="01:30-02:00"
        ),
        Chunk(
            chunk_id="chunk_005",
            title="AI Ethics",
            content="AI ethics is becoming increasingly important. We must consider bias, privacy, and accountability in AI systems.",
            keywords=["ethics", "bias", "privacy", "accountability"],
            named_entities=["AI ethics"],
            timestamp_range="02:00-02:30"
        )
    ]
    return chunks

def demo_chunk_radius(processor, chunks):
    """Demonstrate chunk radius calculation"""
    print("\n" + "="*60)
    print("📏 CHUNK RADIUS DEMONSTRATION")
    print("="*60)
    
    print("\nChunk radius measures how 'clustered' a chunk is with its neighbors.")
    print("Higher radius = more similar to surrounding chunks")
    
    for chunk in chunks:
        try:
            # Calculate radius for each chunk
            radius = processor.chunk_radius(chunk, threshold=0.7, n=3)
            
            print(f"\n🔍 Chunk: {chunk.title}")
            print(f"   ID: {chunk.chunk_id}")
            print(f"   Content: {chunk.content[:80]}...")
            print(f"   Radius Score: {radius:.3f}")
            
            # Interpret the radius
            if radius >= 0.8:
                interpretation = "Highly clustered - very similar to neighbors"
            elif radius >= 0.6:
                interpretation = "Moderately clustered - somewhat similar to neighbors"
            elif radius >= 0.4:
                interpretation = "Loosely clustered - some similarity to neighbors"
            else:
                interpretation = "Isolated - very different from neighbors"
            
            print(f"   Interpretation: {interpretation}")
            
        except Exception as e:
            print(f"❌ Error calculating radius for {chunk.chunk_id}: {e}")

def demo_merge_analysis(processor, chunks):
    """Demonstrate merge analysis between chunks"""
    print("\n" + "="*60)
    print("🔗 CHUNK MERGE ANALYSIS")
    print("="*60)
    
    print("\nAnalyzing which chunks should be merged based on relevance and coherence...")
    
    # Analyze pairs of chunks
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            chunk_a = chunks[i]
            chunk_b = chunks[j]
            
            try:
                # Check if chunks should be merged
                should_merge = processor.should_merge_chunks(
                    chunk_a, 
                    chunk_b, 
                    relevance_threshold=0.7, 
                    coherence_threshold=0.5
                )
                
                print(f"\n🔍 Analyzing: {chunk_a.title} + {chunk_b.title}")
                print(f"   Chunk A: {chunk_a.chunk_id}")
                print(f"   Chunk B: {chunk_b.chunk_id}")
                print(f"   Should Merge: {'✅ YES' if should_merge else '❌ NO'}")
                
                if should_merge:
                    print("   💡 These chunks are overly similar and could be merged!")
                
            except Exception as e:
                print(f"❌ Error analyzing merge for {chunk_a.chunk_id} + {chunk_b.chunk_id}: {e}")

def demo_chunk_merging(processor, chunks):
    """Demonstrate actual chunk merging"""
    print("\n" + "="*60)
    print("🔄 CHUNK MERGING DEMONSTRATION")
    print("="*60)
    
    print("\nDemonstrating chunk merging functionality...")
    
    # Merge the first two chunks (which are likely similar)
    chunk_a = chunks[0]
    chunk_b = chunks[1]
    
    try:
        print(f"\n🔄 Merging chunks:")
        print(f"   Chunk A: {chunk_a.title}")
        print(f"   Content: {chunk_a.content[:60]}...")
        print(f"   Chunk B: {chunk_b.title}")
        print(f"   Content: {chunk_b.content[:60]}...")
        
        # Perform the merge
        merged_chunk = processor.merge_chunks(chunk_a, chunk_b)
        
        print(f"\n✅ Merge Result:")
        print(f"   New ID: {merged_chunk.chunk_id}")
        print(f"   New Title: {merged_chunk.title}")
        print(f"   Combined Content: {merged_chunk.content[:100]}...")
        print(f"   Keywords: {merged_chunk.keywords}")
        print(f"   Named Entities: {merged_chunk.named_entities}")
        print(f"   Metadata: {merged_chunk.metadata}")
        
    except Exception as e:
        print(f"❌ Error merging chunks: {e}")

def demo_coherence_scoring(processor, chunks):
    """Demonstrate coherence scoring between chunks"""
    print("\n" + "="*60)
    print("🎯 COHERENCE SCORING DEMONSTRATION")
    print("="*60)
    
    print("\nCoherence score measures how often chunks are used together.")
    print("Higher coherence = chunks are frequently used in similar contexts.")
    
    # Analyze coherence between chunk pairs
    for i in range(len(chunks)):
        for j in range(i + 1, min(i + 2, len(chunks))):  # Only analyze adjacent pairs
            chunk_a = chunks[i]
            chunk_b = chunks[j]
            
            try:
                coherence = processor._calculate_coherence_score(chunk_a.chunk_id, chunk_b.chunk_id)
                
                print(f"\n🔍 Coherence Analysis:")
                print(f"   Chunk A: {chunk_a.title} ({chunk_a.chunk_id})")
                print(f"   Chunk B: {chunk_b.title} ({chunk_b.chunk_id})")
                print(f"   Coherence Score: {coherence:.3f}")
                
                # Interpret coherence
                if coherence >= 0.8:
                    interpretation = "Very high coherence - chunks used together frequently"
                elif coherence >= 0.6:
                    interpretation = "High coherence - chunks used together often"
                elif coherence >= 0.4:
                    interpretation = "Moderate coherence - some usage together"
                elif coherence >= 0.2:
                    interpretation = "Low coherence - rarely used together"
                else:
                    interpretation = "Very low coherence - almost never used together"
                
                print(f"   Interpretation: {interpretation}")
                
            except Exception as e:
                print(f"❌ Error calculating coherence: {e}")

def main():
    """Main demonstration function"""
    print("🚀 EchoGem Advanced Chunk Management Demo")
    print("="*60)
    
    # Check for API keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not google_api_key:
        print("❌ GOOGLE_API_KEY environment variable not set")
        print("   Get your free API key from: https://makersuite.google.com/app/apikey")
        return
    
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        print("   Get your API key from: https://app.pinecone.io/")
        return
    
    print("✅ API keys found")
    
    try:
        # Initialize processor
        print("\n🔧 Initializing EchoGem processor...")
        processor = Processor(
            google_api_key=google_api_key,
            pinecone_api_key=pinecone_api_key
        )
        print("✅ Processor initialized")
        
        # Create sample chunks
        print("\n📝 Creating sample chunks for demonstration...")
        chunks = create_sample_chunks()
        print(f"✅ Created {len(chunks)} sample chunks")
        
        # Store chunks in vector database for radius calculation
        print("\n💾 Storing chunks in vector database...")
        for chunk in chunks:
            try:
                processor.vector_db.add_chunk(chunk)
                processor.usage_cache.record_chunk_usage(chunk.chunk_id)
            except Exception as e:
                print(f"⚠️  Warning: Could not store chunk {chunk.chunk_id}: {e}")
        
        # Run demonstrations
        demo_chunk_radius(processor, chunks)
        demo_merge_analysis(processor, chunks)
        demo_chunk_merging(processor, chunks)
        demo_coherence_scoring(processor, chunks)
        
        print("\n" + "="*60)
        print("🎉 Advanced Chunk Management Demo Complete!")
        print("="*60)
        
        print("\n📚 Key Features Demonstrated:")
        print("   • Chunk Radius: Measures clustering with neighbors")
        print("   • Merge Analysis: Determines if chunks should be combined")
        print("   • Chunk Merging: Combines overly similar chunks")
        print("   • Coherence Scoring: Measures usage patterns together")
        
        print("\n💡 Use Cases:")
        print("   • Reduce fragmentation in transcript analysis")
        print("   • Improve chunk quality by merging similar content")
        print("   • Understand chunk relationships and clustering")
        print("   • Optimize retrieval by combining related information")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
