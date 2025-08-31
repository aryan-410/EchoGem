#!/usr/bin/env python3
"""
EchoGem Python API Demo

This demo showcases all the Python programming interface features:
- Direct class usage
- Custom chunking strategies
- Advanced querying
- Data manipulation
- Integration examples
"""

import os
import time
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from processor import Processor
from chunker import Chunker
from vector_store import ChunkVectorDB
from prompt_answer_store import PromptAnswerVectorDB
from usage_cache import UsageCache
from models import (
    Chunk, ChunkResponse, PAPair, QueryResult,
    ChunkingOptions, QueryOptions
)

def create_sample_data():
    """Create various types of sample data for demonstration"""
    samples = {
        "academic_lecture": """
        Lecture: Introduction to Quantum Computing
        Professor: Dr. Elena Rodriguez
        Date: March 15, 2024
        
        Today we'll explore the fundamental principles of quantum computing. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in superposition states.
        
        The key principles we'll cover are:
        1. Superposition: A qubit can be in multiple states simultaneously
        2. Entanglement: Qubits can be correlated in ways classical bits cannot
        3. Interference: Quantum states can interfere constructively or destructively
        
        Let's start with superposition. Imagine a coin that's spinning. While it's spinning, it's neither heads nor tails - it's in a superposition of both states. When we measure it, the superposition collapses to one definite state.
        
        In quantum computing, we can create algorithms that exploit superposition to perform certain calculations exponentially faster than classical computers. Shor's algorithm for factoring large numbers is a famous example.
        
        However, quantum computers are not universally faster. They excel at specific problems like:
        - Factoring large numbers
        - Searching unsorted databases
        - Simulating quantum systems
        - Optimization problems
        
        The main challenges in building quantum computers are:
        - Decoherence: Quantum states are fragile and easily disturbed
        - Error correction: We need sophisticated error correction codes
        - Scalability: Adding more qubits increases complexity exponentially
        
        Despite these challenges, we've made remarkable progress. IBM has built quantum computers with over 100 qubits, and Google achieved quantum supremacy in 2019.
        
        Next week, we'll dive deeper into quantum algorithms and see how they work in practice.
        """,
        
        "business_presentation": """
        Q1 2024 Business Review
        Presenter: Sarah Johnson, CEO
        Date: April 10, 2024
        
        Good morning everyone. Let me start by thanking the entire team for an outstanding Q1 performance. We've exceeded our targets across all key metrics.
        
        Financial Highlights:
        - Revenue: $12.4M (up 23% from Q4 2023)
        - Profit margin: 18.7% (up 2.1 percentage points)
        - Customer acquisition cost: $45 (down 15%)
        - Customer lifetime value: $890 (up 12%)
        
        Product Performance:
        Our flagship product, DataFlow Pro, saw 34% growth in active users. The new AI-powered analytics features launched in February have been particularly well-received, with 78% adoption rate among enterprise customers.
        
        The mobile app redesign resulted in:
        - 45% increase in daily active users
        - 28% improvement in session duration
        - 67% reduction in crash rate
        
        Market Expansion:
        We successfully entered three new markets:
        1. Southeast Asia - 15% of Q1 revenue
        2. Latin America - 12% of Q1 revenue
        3. Eastern Europe - 8% of Q1 revenue
        
        Team Growth:
        We've added 23 new team members across engineering, sales, and customer success. Our employee satisfaction score is 4.8/5, up from 4.6 last quarter.
        
        Challenges and Opportunities:
        The main challenge is scaling our infrastructure to handle the increased load. We're investing $2M in cloud infrastructure upgrades this quarter.
        
        Opportunities include:
        - Expanding our AI capabilities
        - Developing industry-specific solutions
        - Building strategic partnerships
        
        Q2 Goals:
        - Achieve $15M revenue target
        - Launch enterprise security features
        - Expand to two additional markets
        - Maintain 4.8+ employee satisfaction score
        
        I'm confident we can achieve these goals with the incredible team we have.
        """,
        
        "technical_documentation": """
        EchoGem API Reference
        Version: 1.0.0
        Last Updated: April 2024
        
        Overview:
        EchoGem is a Python library for intelligent transcript processing and question answering. It uses advanced NLP techniques to chunk transcripts semantically and provides a powerful query interface.
        
        Core Components:
        
        1. Processor Class
        The main interface for transcript processing and querying.
        
        Methods:
        - process_transcript(text, options): Process and chunk a transcript
        - query(question, options): Query the knowledge base
        - get_statistics(): Get processing statistics
        
        2. Chunker Class
        Handles text chunking with various strategies.
        
        Methods:
        - chunk_text(text, options): Create semantic chunks
        - analyze_chunks(chunks): Analyze chunk quality and relationships
        
        3. Vector Store Classes
        Manage storage and retrieval of chunks and Q&A pairs.
        
        ChunkVectorDB:
        - add_chunks(chunks): Store new chunks
        - search_chunks(query, limit): Find relevant chunks
        - get_chunk(id): Retrieve specific chunk
        
        PromptAnswerVectorDB:
        - add_pa_pair(pair): Store Q&A pair
        - search_pa_pairs(query, limit): Find relevant Q&A pairs
        
        4. Usage Cache
        Tracks how chunks and Q&A pairs are used.
        
        Methods:
        - record_chunk_access(chunk_id): Log chunk usage
        - get_usage_statistics(): Get usage analytics
        - export_usage_data(): Export usage data
        
        Configuration:
        Environment variables:
        - GOOGLE_API_KEY: Required for Gemini integration
        - PINECONE_API_KEY: Required for vector storage
        - PINECONE_ENVIRONMENT: Pinecone environment
        
        Example Usage:
        
        from echogem import Processor, ChunkingOptions, QueryOptions
        
        # Initialize processor
        processor = Processor()
        
        # Process transcript
        options = ChunkingOptions(max_chunk_size=500, overlap=50)
        response = processor.process_transcript(transcript_text, options)
        
        # Query knowledge base
        query_opts = QueryOptions(show_chunks=True, max_chunks=3)
        result = processor.query("What is the main topic?", query_opts)
        
        print(result.answer)
        print(f"Used {len(result.chunks_used)} chunks")
        """
    }
    
    return samples

def demo_basic_api():
    """Demonstrate basic API usage"""
    print("🔧 EchoGem Python API Demo")
    print("=" * 40)
    print("This demo will show you how to use EchoGem programmatically!")
    print()
    
    # Initialize components
    print("1️⃣ Initializing EchoGem components...")
    try:
        processor = Processor()
        chunker = Chunker()
        chunk_db = ChunkVectorDB()
        qa_db = PromptAnswerVectorDB()
        usage_cache = UsageCache()
        print("   ✅ All components initialized successfully")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        print("   💡 Make sure your API keys are set correctly")
        return None
    
    # Create sample data
    print("\n2️⃣ Creating sample data...")
    samples = create_sample_data()
    for name, content in samples.items():
        print(f"   📝 {name}: {len(content)} characters")
    
    return processor, chunker, chunk_db, qa_db, usage_cache, samples

def demo_chunking_strategies(processor, chunker, samples):
    """Demonstrate different chunking strategies"""
    print("\n3️⃣ Chunking Strategies Demo")
    print("=" * 35)
    
    # Test different chunking options
    chunking_configs = [
        ("Small chunks", ChunkingOptions(max_chunk_size=200, overlap=25)),
        ("Medium chunks", ChunkingOptions(max_chunk_size=400, overlap=50)),
        ("Large chunks", ChunkingOptions(max_chunk_size=800, overlap=100)),
        ("Semantic chunks", ChunkingOptions(max_chunk_size=500, overlap=75, semantic_chunking=True))
    ]
    
    results = {}
    for config_name, options in chunking_configs:
        print(f"\n   🔧 Testing: {config_name}")
        print(f"      Options: size={options.max_chunk_size}, overlap={options.overlap}, semantic={options.semantic_chunking}")
        
        # Test with academic lecture
        text = samples["academic_lecture"]
        start_time = time.time()
        
        try:
            if options.semantic_chunking:
                response = processor.process_transcript(text, chunking_options=options)
                chunks = response.chunks
            else:
                chunks = chunker.chunk_text(text, options)
            
            processing_time = time.time() - start_time
            
            print(f"      ✅ Created {len(chunks)} chunks in {processing_time:.2f}s")
            print(f"      📊 Average chunk size: {sum(len(c.content) for c in chunks) // len(chunks)} chars")
            
            # Analyze chunk quality
            if hasattr(chunker, 'analyze_chunks'):
                analysis = chunker.analyze_chunks(chunks)
                print(f"      🎯 Quality score: {analysis.get('overall_quality', 'N/A')}")
            
            results[config_name] = {
                'chunks': chunks,
                'time': processing_time,
                'count': len(chunks)
            }
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            results[config_name] = None
    
    return results

def demo_query_interface(processor, samples):
    """Demonstrate the query interface"""
    print("\n4️⃣ Query Interface Demo")
    print("=" * 30)
    
    # Process a sample for querying
    print("   📝 Processing sample for querying...")
    try:
        response = processor.process_transcript(
            samples["business_presentation"],
            chunking_options=ChunkingOptions(max_chunk_size=400, overlap=50)
        )
        print(f"   ✅ Created {len(response.chunks)} chunks for querying")
    except Exception as e:
        print(f"   ❌ Processing failed: {e}")
        return
    
    # Test different query types
    queries = [
        ("Financial performance", "What were the key financial metrics in Q1?"),
        ("Product success", "How did the mobile app redesign perform?"),
        ("Market expansion", "Which new markets did the company enter?"),
        ("Team growth", "How has the team grown and what's the satisfaction score?"),
        ("Future plans", "What are the goals for Q2?")
    ]
    
    query_configs = [
        ("Basic query", QueryOptions()),
        ("Show chunks", QueryOptions(show_chunks=True, max_chunks=2)),
        ("Show Q&A pairs", QueryOptions(show_prompt_answers=True, max_chunks=3)),
        ("Comprehensive", QueryOptions(show_chunks=True, show_prompt_answers=True, max_chunks=5))
    ]
    
    for query_name, query_text in queries:
        print(f"\n   🤔 {query_name}: {query_text}")
        
        for config_name, options in query_configs:
            print(f"      🔧 {config_name}:")
            
            start_time = time.time()
            try:
                result = processor.query(query_text, query_options=options)
                query_time = time.time() - start_time
                
                print(f"         ⏱️  Response time: {query_time:.2f}s")
                print(f"         📝 Answer: {result.answer[:80]}...")
                
                if hasattr(result, 'chunks_used') and result.chunks_used:
                    print(f"         🔗 Chunks used: {len(result.chunks_used)}")
                
                if hasattr(result, 'prompt_answers_used') and result.prompt_answers_used:
                    print(f"         💬 Q&A pairs used: {len(result.prompt_answers_used)}")
                
            except Exception as e:
                print(f"         ❌ Query failed: {e}")

def demo_advanced_features(processor, chunk_db, qa_db, usage_cache):
    """Demonstrate advanced features"""
    print("\n5️⃣ Advanced Features Demo")
    print("=" * 35)
    
    # Custom chunk analysis
    print("\n   🔍 Custom chunk analysis...")
    try:
        # Get some chunks from the database
        chunks = chunk_db.search_chunks("quantum computing", limit=5)
        if chunks:
            print(f"      📦 Found {len(chunks)} chunks about quantum computing")
            
            # Analyze chunk relationships
            for i, chunk in enumerate(chunks[:3]):
                print(f"         Chunk {i+1}: {chunk.title[:50]}...")
                print(f"            Keywords: {', '.join(chunk.keywords[:3])}")
                print(f"            Entities: {', '.join(chunk.entities[:3])}")
        else:
            print("      ⚠️  No chunks found for analysis")
    except Exception as e:
        print(f"      ❌ Chunk analysis failed: {e}")
    
    # Usage analytics
    print("\n   📊 Usage analytics...")
    try:
        stats = usage_cache.get_usage_statistics()
        print(f"      📈 Total chunks accessed: {stats.get('total_chunks_accessed', 0)}")
        print(f"      🔥 Most used chunks: {len(stats.get('most_used_chunks', []))}")
        print(f"      ⏰ Recent activity: {len(stats.get('recent_activity', []))}")
        
        # Show recent activity
        recent = stats.get('recent_activity', [])
        if recent:
            print("      🕐 Recent chunk accesses:")
            for activity in recent[:3]:
                chunk_id = activity.get('chunk_id', 'Unknown')[:8]
                timestamp = activity.get('timestamp', 'Unknown')
                print(f"         {chunk_id} at {timestamp}")
                
    except Exception as e:
        print(f"      ❌ Usage analytics failed: {e}")
    
    # Custom Q&A pair creation
    print("\n   💬 Custom Q&A pair creation...")
    try:
        custom_qa = PAPair(
            question="What are the main challenges in quantum computing?",
            answer="The main challenges include decoherence, error correction, and scalability issues.",
            metadata={
                "source": "demo",
                "confidence": 0.95,
                "created_at": datetime.now().isoformat()
            }
        )
        
        qa_db.add_pa_pair(custom_qa)
        print("      ✅ Custom Q&A pair created and stored")
        
        # Test retrieval
        retrieved = qa_db.search_pa_pairs("quantum computing challenges", limit=1)
        if retrieved:
            print(f"      🔍 Retrieved: {retrieved[0].question[:60]}...")
        
    except Exception as e:
        print(f"      ❌ Q&A pair creation failed: {e}")

def demo_integration_examples(processor):
    """Show integration examples"""
    print("\n6️⃣ Integration Examples")
    print("=" * 30)
    
    # Example 1: Batch processing
    print("\n   📦 Batch processing example...")
    try:
        # Simulate batch processing
        documents = [
            ("doc1", "This is the first document about machine learning."),
            ("doc2", "The second document covers deep learning and neural networks."),
            ("doc3", "Document three discusses natural language processing.")
        ]
        
        all_chunks = []
        for doc_id, content in documents:
            response = processor.process_transcript(
                content,
                chunking_options=ChunkingOptions(max_chunk_size=100, overlap=20)
            )
            all_chunks.extend(response.chunks)
            print(f"         Processed {doc_id}: {len(response.chunks)} chunks")
        
        print(f"      📊 Total chunks across all documents: {len(all_chunks)}")
        
    except Exception as e:
        print(f"      ❌ Batch processing failed: {e}")
    
    # Example 2: Custom scoring
    print("\n   🎯 Custom scoring example...")
    try:
        # Simulate custom scoring logic
        query = "machine learning applications"
        chunks = chunk_db.search_chunks(query, limit=10)
        
        if chunks:
            # Custom scoring based on content length and keyword density
            scored_chunks = []
            for chunk in chunks:
                # Simple scoring: longer content + more keywords = higher score
                keyword_score = len(chunk.keywords) * 0.1
                length_score = min(len(chunk.content) / 1000, 1.0) * 0.5
                total_score = keyword_score + length_score
                
                scored_chunks.append((chunk, total_score))
            
            # Sort by score
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            print(f"      🏆 Top 3 chunks by custom score:")
            for i, (chunk, score) in enumerate(scored_chunks[:3]):
                print(f"         {i+1}. Score {score:.3f}: {chunk.title[:40]}...")
        
    except Exception as e:
        print(f"      ❌ Custom scoring failed: {e}")

def demo_performance_metrics(processor, samples):
    """Demonstrate performance metrics"""
    print("\n7️⃣ Performance Metrics")
    print("=" * 30)
    
    # Test processing performance
    print("\n   ⚡ Processing performance...")
    test_text = samples["technical_documentation"]
    
    chunk_sizes = [200, 400, 600, 800]
    results = {}
    
    for size in chunk_sizes:
        options = ChunkingOptions(max_chunk_size=size, overlap=size//4)
        
        # Time the processing
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            response = processor.process_transcript(test_text, chunking_options=options)
            processing_time = time.time() - start_time
            end_memory = get_memory_usage()
            
            results[size] = {
                'time': processing_time,
                'chunks': len(response.chunks),
                'memory_delta': end_memory - start_memory if start_memory and end_memory else 0
            }
            
            print(f"      📏 Chunk size {size}: {processing_time:.3f}s, {len(response.chunks)} chunks")
            
        except Exception as e:
            print(f"      ❌ Size {size} failed: {e}")
    
    # Performance analysis
    if results:
        print("\n      📊 Performance Analysis:")
        fastest = min(results.items(), key=lambda x: x[1]['time'])
        most_efficient = min(results.items(), key=lambda x: x[1]['time'] / x[1]['chunks'])
        
        print(f"         🏃 Fastest: {fastest[0]} chars ({fastest[1]['time']:.3f}s)")
        print(f"         ⚡ Most efficient: {most_efficient[0]} chars ({most_efficient[1]['time']/most_efficient[1]['chunks']:.3f}s per chunk)")

def get_memory_usage():
    """Get current memory usage (if psutil is available)"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        return None

def main():
    """Main API demo function"""
    print("🎯 EchoGem Python API Demonstration")
    print("=" * 60)
    print("This demo will show you how to use EchoGem programmatically!")
    print()
    
    # Run basic API demo
    result = demo_basic_api()
    if not result:
        print("\n❌ API demo failed. Please check your setup and try again.")
        return
    
    processor, chunker, chunk_db, qa_db, usage_cache, samples = result
    
    # Run all demo sections
    chunking_results = demo_chunking_strategies(processor, chunker, samples)
    demo_query_interface(processor, samples)
    demo_advanced_features(processor, chunk_db, qa_db, usage_cache)
    demo_integration_examples(processor)
    demo_performance_metrics(processor, samples)
    
    # Final recommendations
    print("\n🎉 API Demo Complete!")
    print("=" * 25)
    print("💡 Key API patterns to remember:")
    print("   🔧 Initialize: processor = Processor()")
    print("   📝 Process: response = processor.process_transcript(text, options)")
    print("   ❓ Query: result = processor.query(question, options)")
    print("   📊 Stats: stats = usage_cache.get_usage_statistics()")
    print("   🔍 Search: chunks = chunk_db.search_chunks(query, limit=5)")
    
    print("\n📚 Explore other demos:")
    print("   - Basic workflow: python demos/01_basic_workflow_demo.py")
    print("   - CLI usage: python demos/02_cli_demo.py")
    print("   - Performance: python demos/09_performance_benchmarking_demo.py")

if __name__ == "__main__":
    main()
