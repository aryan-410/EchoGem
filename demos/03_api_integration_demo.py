#!/usr/bin/env python3
"""
EchoGem API Integration Demo
============================

This demo showcases programmatic usage of the EchoGem library.
Learn how to integrate EchoGem into your applications, handle errors,
and optimize performance through direct API calls.

Prerequisites:
- Set GOOGLE_API_KEY environment variable
- Set PINECONE_API_KEY environment variable
- Have sample transcript files ready
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the parent directory to Python path to import echogem
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_environment():
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

def demo_basic_initialization():
    """Demonstrate basic processor initialization"""
    print("\n" + "="*60)
    print("BASIC INITIALIZATION")
    print("="*60)

    try:
        from echogem.processor import Processor
        
        print("1. Basic initialization:")
        processor = Processor()
        print("   ✓ Default processor created successfully")
        
        print("\n2. Custom configuration:")
        custom_processor = Processor(
            chunk_index_name="custom-chunks",
            pa_index_name="custom-qa",
            embedding_model="all-MiniLM-L6-v2",
            usage_cache_path="custom_cache.csv"
        )
        print("   ✓ Custom processor created successfully")
        
        print("\n3. Available attributes:")
        print(f"   - Chunk index: {custom_processor.chunk_index_name}")
        print(f"   - QA index: {custom_processor.pa_index_name}")
        print(f"   - Embedding model: {custom_processor.embedding_model}")
        print(f"   - Cache path: {custom_processor.usage_cache_path}")
        
        return processor, custom_processor
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return None, None

def demo_transcript_processing(processor):
    """Demonstrate transcript processing with different options"""
    print("\n" + "="*60)
    print("TRANSCRIPT PROCESSING")
    print("="*60)

    if not processor:
        print("❌ Processor not available, skipping this demo")
        return False

    # Check if sample transcript exists
    transcript_path = Path("../examples/sample_transcript.txt")
    if not transcript_path.exists():
        print(f"❌ Sample transcript not found: {transcript_path}")
        return False

    try:
        from echogem.models import ChunkingOptions
        
        print("1. Basic processing:")
        start_time = time.time()
        response = processor.chunk_and_process(str(transcript_path))
        processing_time = time.time() - start_time
        
        if response.success:
            print(f"   ✓ Successfully processed {response.num_chunks} chunks")
            print(f"   ✓ Processing time: {processing_time:.2f} seconds")
        else:
            print(f"   ❌ Processing failed: {response.error_message}")
            return False

        print("\n2. Processing with custom options:")
        custom_options = ChunkingOptions(
            chunk_size=800,
            overlap=150,
            semantic_chunking=True,
            max_tokens=3000,
            similarity_threshold=0.8,
            coherence_threshold=0.7,
            show_chunks=True,
            show_metadata=True
        )
        
        start_time = time.time()
        custom_response = processor.chunk_and_process(
            str(transcript_path), 
            options=custom_options,
            output_chunks=True
        )
        custom_time = time.time() - start_time
        
        if custom_response.success:
            print(f"   ✓ Custom processing: {custom_response.num_chunks} chunks")
            print(f"   ✓ Custom time: {custom_time:.2f} seconds")
            
            if hasattr(custom_response, 'chunks') and custom_response.chunks:
                print(f"   ✓ Chunk details available")
                for i, chunk in enumerate(custom_response.chunks[:2], 1):
                    print(f"     Chunk {i}: {chunk.title}")
                    print(f"       Content: {chunk.content[:100]}...")
        else:
            print(f"   ❌ Custom processing failed: {custom_response.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        traceback.print_exc()
        return False

def demo_question_answering(processor):
    """Demonstrate question answering with different configurations"""
    print("\n" + "="*60)
    print("QUESTION ANSWERING")
    print("="*60)

    if not processor:
        print("❌ Processor not available, skipping this demo")
        return False

    try:
        from echogem.models import QueryOptions
        
        questions = [
            "What is the main topic of this meeting?",
            "Who are the participants and what are their roles?",
            "What are the key concerns about AI implementation?",
            "What is the timeline for FDA approval?",
            "What security measures are being implemented?"
        ]
        
        print("1. Basic question answering:")
        for i, question in enumerate(questions[:2], 1):
            print(f"\n   Q{i}: {question}")
            start_time = time.time()
            result = processor.answer_question(question)
            query_time = time.time() - start_time
            
            print(f"   A: {result.answer}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Query time: {query_time:.2f}s")
            print(f"   Chunks used: {len(result.chunks_used)}")

        print("\n2. Advanced question answering with custom options:")
        custom_query_opts = QueryOptions(
            top_k=8,
            similarity_threshold=0.6,
            include_metadata=True,
            max_tokens=4000,
            temperature=0.1,
            use_cache=True,
            show_chunks=True,
            show_metadata=True
        )
        
        advanced_question = "What are the key decisions and action items from this meeting?"
        print(f"\n   Advanced Q: {advanced_question}")
        
        start_time = time.time()
        advanced_result = processor.answer_question(advanced_question, options=custom_query_opts)
        advanced_time = time.time() - start_time
        
        print(f"   A: {advanced_result.answer}")
        print(f"   Confidence: {advanced_result.confidence:.2f}")
        print(f"   Advanced time: {advanced_time:.2f}s")
        print(f"   Chunks used: {len(advanced_result.chunks_used)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Question answering error: {e}")
        traceback.print_exc()
        return False

def demo_direct_chunk_access(processor):
    """Demonstrate direct access to chunks and Q&A pairs"""
    print("\n" + "="*60)
    print("DIRECT CHUNK ACCESS")
    print("="*60)

    if not processor:
        print("❌ Processor not available, skipping this demo")
        return False

    try:
        print("1. Retrieving chunks for specific queries:")
        chunk_queries = [
            "AI implementation",
            "security measures",
            "FDA approval process"
        ]
        
        for query in chunk_queries:
            print(f"\n   Query: '{query}'")
            chunks = processor.pick_chunks(
                query=query,
                k=3,
                entropy_weight=0.3,
                recency_weight=0.2
            )
            
            print(f"   Found {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks, 1):
                print(f"     {i}. {chunk.title}")
                print(f"        Score: {chunk.score:.3f}")
                print(f"        Content: {chunk.content[:80]}...")

        print("\n2. Finding similar Q&A pairs:")
        qa_queries = [
            "What is the main topic?",
            "Who are the speakers?",
            "What decisions were made?"
        ]
        
        for query in qa_queries:
            print(f"\n   Query: '{query}'")
            qa_pairs = processor.get_similar_qa_pairs(
                query=query,
                k=2,
                sim_weight=0.6,
                entropy_weight=0.2,
                recency_weight=0.2
            )
            
            print(f"   Found {len(qa_pairs)} Q&A pairs:")
            for i, qa in enumerate(qa_pairs, 1):
                print(f"     {i}. Q: {qa.prompt}")
                print(f"        A: {qa.answer}")
                print(f"        Similarity: {qa.similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct access error: {e}")
        traceback.print_exc()
        return False

def demo_error_handling():
    """Demonstrate comprehensive error handling"""
    print("\n" + "="*60)
    print("ERROR HANDLING")
    print("="*60)

    try:
        from echogem.processor import Processor
        
        print("1. Handling missing API keys:")
        try:
            # Temporarily unset API keys
            original_google_key = os.environ.get("GOOGLE_API_KEY")
            original_pinecone_key = os.environ.get("PINECONE_API_KEY")
            
            if "GOOGLE_API_KEY" in os.environ:
                del os.environ["GOOGLE_API_KEY"]
            if "PINECONE_API_KEY" in os.environ:
                del os.environ["PINECONE_API_KEY"]
            
            try:
                processor = Processor()
                print("   ❌ Should have failed without API keys")
            except Exception as e:
                print(f"   ✓ Correctly caught error: {type(e).__name__}")
                print(f"   ✓ Error message: {str(e)}")
            
            # Restore API keys
            if original_google_key:
                os.environ["GOOGLE_API_KEY"] = original_google_key
            if original_pinecone_key:
                os.environ["PINECONE_API_KEY"] = original_pinecone_key
                
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")

        print("\n2. Handling invalid file paths:")
        try:
            processor = Processor()
            response = processor.chunk_and_process("nonexistent_file.txt")
            if not response.success:
                print(f"   ✓ Correctly handled missing file: {response.error_message}")
            else:
                print("   ❌ Should have failed with missing file")
        except Exception as e:
            print(f"   ✓ Caught exception: {type(e).__name__}")

        print("\n3. Handling invalid parameters:")
        try:
            from echogem.models import ChunkingOptions
            invalid_options = ChunkingOptions(
                chunk_size=-100,  # Invalid negative size
                overlap=1000      # Overlap larger than chunk size
            )
            print("   ✓ Invalid options created (validation will occur during processing)")
        except Exception as e:
            print(f"   ✓ Caught validation error: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling demo failed: {e}")
        traceback.print_exc()
        return False

def demo_performance_optimization(processor):
    """Demonstrate performance optimization techniques"""
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION")
    print("="*60)

    if not processor:
        print("❌ Processor not available, skipping this demo")
        return False

    try:
        print("1. Performance benchmarking:")
        
        # Test different chunk sizes
        chunk_sizes = [500, 800, 1000, 1200]
        results = {}
        
        transcript_path = Path("../examples/sample_transcript.txt")
        if not transcript_path.exists():
            print("   ❌ Sample transcript not found, skipping performance test")
            return False
        
        for size in chunk_sizes:
            print(f"\n   Testing chunk size: {size}")
            start_time = time.time()
            
            from echogem.models import ChunkingOptions
            options = ChunkingOptions(chunk_size=size, overlap=size//4)
            
            response = processor.chunk_and_process(str(transcript_path), options=options)
            processing_time = time.time() - start_time
            
            if response.success:
                results[size] = {
                    'time': processing_time,
                    'chunks': response.num_chunks,
                    'efficiency': response.num_chunks / processing_time
                }
                print(f"     ✓ Time: {processing_time:.2f}s, Chunks: {response.num_chunks}")
            else:
                print(f"     ❌ Failed: {response.error_message}")
        
        print("\n2. Performance analysis:")
        if results:
            best_size = max(results.keys(), key=lambda k: results[k]['efficiency'])
            print(f"   Most efficient chunk size: {best_size}")
            print(f"   Efficiency: {results[best_size]['efficiency']:.2f} chunks/second")
            
            print("\n   Performance comparison:")
            for size, metrics in results.items():
                print(f"     {size}: {metrics['time']:.2f}s, {metrics['chunks']} chunks, "
                      f"{metrics['efficiency']:.2f} chunks/s")

        print("\n3. Query performance optimization:")
        test_questions = [
            "What is the main topic?",
            "Who are the speakers?",
            "What decisions were made?"
        ]
        
        # Test different similarity thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8]
        query_results = {}
        
        for threshold in thresholds:
            print(f"\n   Testing similarity threshold: {threshold}")
            total_time = 0
            successful_queries = 0
            
            for question in test_questions:
                try:
                    from echogem.models import QueryOptions
                    options = QueryOptions(similarity_threshold=threshold, top_k=3)
                    
                    start_time = time.time()
                    result = processor.answer_question(question, options=options)
                    query_time = time.time() - start_time
                    
                    total_time += query_time
                    successful_queries += 1
                    
                except Exception as e:
                    print(f"     ❌ Query failed: {e}")
            
            if successful_queries > 0:
                avg_time = total_time / successful_queries
                query_results[threshold] = avg_time
                print(f"     ✓ Average query time: {avg_time:.2f}s")
        
        if query_results:
            best_threshold = min(query_results.keys(), key=lambda k: query_results[k])
            print(f"\n   Optimal similarity threshold: {best_threshold}")
            print(f"   Best average query time: {query_results[best_threshold]:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization error: {e}")
        traceback.print_exc()
        return False

def demo_system_statistics(processor):
    """Demonstrate system statistics and monitoring"""
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)

    if not processor:
        print("❌ Processor not available, skipping this demo")
        return False

    try:
        print("1. Basic system statistics:")
        stats = processor.get_stats()
        
        if stats:
            print("   System Overview:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"     {key}: {value:.2f}")
                else:
                    print(f"     {key}: {value}")
        else:
            print("   ❌ No statistics available")

        print("\n2. Usage cache statistics:")
        try:
            from echogem.usage_cache import UsageCache
            usage_cache = UsageCache()
            usage_stats = usage_cache.get_usage_statistics()
            
            if usage_stats:
                print("   Usage Analytics:")
                for key, value in usage_stats.items():
                    if isinstance(value, float):
                        print(f"     {key}: {value:.2f}")
                    else:
                        print(f"     {key}: {value}")
            else:
                print("   ❌ No usage statistics available")
                
        except Exception as e:
            print(f"   ⚠️  Usage cache error: {e}")

        print("\n3. Export capabilities:")
        try:
            # Test exporting usage data
            export_path = "demo_usage_export.csv"
            success = usage_cache.export_usage_data(export_path)
            
            if success and Path(export_path).exists():
                print(f"   ✓ Usage data exported to: {export_path}")
                # Clean up
                Path(export_path).unlink()
            else:
                print("   ❌ Usage data export failed")
                
        except Exception as e:
            print(f"   ⚠️  Export error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ System statistics error: {e}")
        traceback.print_exc()
        return False

def demo_integration_patterns():
    """Demonstrate common integration patterns"""
    print("\n" + "="*60)
    print("INTEGRATION PATTERS")
    print("="*60)

    print("1. Web Application Integration:")
    web_app_code = '''
from flask import Flask, request, jsonify
from echogem.processor import Processor

app = Flask(__name__)
processor = Processor()

@app.route('/process', methods=['POST'])
def process_transcript():
    file_path = request.json['file_path']
    response = processor.chunk_and_process(file_path)
    return jsonify({
        'success': response.success,
        'chunks': response.num_chunks,
        'processing_time': response.processing_time
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']
    result = processor.answer_question(question)
    return jsonify({
        'answer': result.answer,
        'confidence': result.confidence,
        'chunks_used': len(result.chunks_used)
    })
'''
    print("   Flask integration example:")
    print("   " + web_app_code.replace('\n', '\n   '))

    print("\n2. Batch Processing Integration:")
    batch_code = '''
import os
from pathlib import Path
from echogem.processor import Processor

def batch_process_transcripts(directory_path):
    processor = Processor()
    results = {}
    
    for file_path in Path(directory_path).glob("*.txt"):
        try:
            response = processor.chunk_and_process(str(file_path))
            results[file_path.name] = {
                'success': response.success,
                'chunks': response.num_chunks if response.success else 0,
                'error': response.error_message if not response.success else None
            }
        except Exception as e:
            results[file_path.name] = {'success': False, 'error': str(e)}
    
    return results
'''
    print("   Batch processing example:")
    print("   " + batch_code.replace('\n', '\n   '))

    print("\n3. Data Pipeline Integration:")
    pipeline_code = '''
from echogem.processor import Processor
from echogem.models import ChunkingOptions, QueryOptions

class EchoGemPipeline:
    def __init__(self):
        self.processor = Processor()
        self.chunking_options = ChunkingOptions(
            chunk_size=1000,
            overlap=200,
            semantic_chunking=True
        )
        self.query_options = QueryOptions(
            top_k=5,
            similarity_threshold=0.7
        )
    
    def process_and_analyze(self, transcript_path, questions):
        # Process transcript
        response = self.processor.chunk_and_process(
            transcript_path, 
            options=self.chunking_options
        )
        
        if not response.success:
            return {'error': response.error_message}
        
        # Analyze with questions
        results = {}
        for question in questions:
            try:
                result = self.processor.answer_question(
                    question, 
                    options=self.query_options
                )
                results[question] = {
                    'answer': result.answer,
                    'confidence': result.confidence
                }
            except Exception as e:
                results[question] = {'error': str(e)}
        
        return {
            'chunks_processed': response.num_chunks,
            'analysis_results': results
        }
'''
    print("   Data pipeline example:")
    print("   " + pipeline_code.replace('\n', '\n   '))

def main():
    """Main demo function"""
    print("🚀 EchoGem API Integration Demo")
    print("="*60)

    # Check environment
    if not check_environment():
        print("\n⚠️  Please set the required environment variables first.")
        print("   You can still view the API examples below.")

    # Run all API demonstrations
    processor, custom_processor = demo_basic_initialization()
    
    if processor:
        demo_transcript_processing(processor)
        demo_question_answering(processor)
        demo_direct_chunk_access(processor)
        demo_performance_optimization(processor)
        demo_system_statistics(processor)
    
    demo_error_handling()
    demo_integration_patterns()

    print("\n" + "="*60)
    print("API INTEGRATION DEMO COMPLETE!")
    print("="*60)
    
    print("\n🎯 What you've learned:")
    print("   ✓ Direct API usage and configuration")
    print("   ✓ Error handling and exception management")
    print("   ✓ Performance optimization techniques")
    print("   ✓ System monitoring and statistics")
    print("   ✓ Common integration patterns")

    print("\n💡 Next steps:")
    print("   1. Integrate EchoGem into your applications")
    print("   2. Implement custom error handling")
    print("   3. Optimize performance for your use case")
    print("   4. Build custom pipelines and workflows")
    print("   5. Check out other demo files for more examples")

    print("\n📚 For more information:")
    print("   - API Reference: ../docs/API_REFERENCE.md")
    print("   - User Guide: ../docs/USER_GUIDE.md")
    print("   - Architecture: ../docs/ARCHITECTURE.md")

if __name__ == "__main__":
    main()
