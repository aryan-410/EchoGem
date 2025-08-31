#!/usr/bin/env python3
"""
EchoGem Performance Benchmarking Demo

This demo showcases comprehensive performance analysis:
- Processing speed benchmarks
- Memory usage analysis
- Chunking efficiency metrics
- Query performance testing
- Optimization recommendations
"""

import os
import time
import json
import gc
import psutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import statistics

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from processor import Processor
from models import ChunkingOptions, QueryOptions

def create_benchmark_data():
    """Create various types of data for benchmarking"""
    data_sets = {
        "small_text": "This is a small text for basic testing. It contains basic information about machine learning and artificial intelligence.",
        
        "medium_text": """
        Machine learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models that enable computers to perform tasks without explicit programming. These algorithms build mathematical models based on sample data, known as training data, to make predictions or decisions without being explicitly programmed to perform the task.
        
        Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. These neural networks are inspired by the structure and function of the human brain, with interconnected nodes that process information and learn from examples.
        
        Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language. NLP combines computational linguistics with statistical, machine learning, and deep learning models to enable computers to process and analyze large amounts of natural language data.
        """,
        
        "large_text": """
        Artificial Intelligence (AI) represents one of the most transformative technologies of the 21st century, fundamentally changing how we live, work, and interact with the world around us. At its core, AI refers to the development of computer systems capable of performing tasks that typically require human intelligence, including learning, reasoning, problem-solving, perception, and language understanding.
        
        The field of AI has evolved significantly since its inception in the 1950s, progressing from simple rule-based systems to sophisticated machine learning algorithms and neural networks. Early AI research focused on symbolic reasoning and expert systems, which attempted to encode human knowledge and reasoning processes into computer programs. While these approaches showed promise in specific domains, they proved limited in their ability to handle the complexity and uncertainty inherent in real-world problems.
        
        The emergence of machine learning in the 1980s and 1990s marked a fundamental shift in AI development. Rather than attempting to explicitly program computers with rules and knowledge, machine learning algorithms learn patterns and relationships from data, improving their performance over time through experience. This approach has proven remarkably successful across a wide range of applications, from image recognition and natural language processing to recommendation systems and autonomous vehicles.
        
        Deep learning, a subset of machine learning that uses artificial neural networks with multiple layers, has been particularly transformative. These neural networks are inspired by the structure and function of the human brain, with interconnected nodes that process information and learn from examples. Deep learning has achieved breakthrough performance in areas such as computer vision, speech recognition, and natural language understanding, often surpassing human performance on specific tasks.
        
        The success of deep learning has been driven by several factors, including the availability of large datasets, advances in computational power, and improvements in algorithm design. The ImageNet dataset, containing millions of labeled images, played a crucial role in advancing computer vision research by providing a standardized benchmark for evaluating algorithm performance. Similarly, the availability of massive text corpora has enabled significant progress in natural language processing.
        
        However, the current state of AI also faces significant challenges and limitations. One of the most pressing concerns is the issue of interpretability and explainability. Many modern AI systems, particularly deep learning models, operate as "black boxes" that make decisions without providing clear explanations for their reasoning. This lack of transparency raises important questions about trust, accountability, and the ability to identify and correct errors or biases.
        
        Another major challenge is the issue of bias and fairness in AI systems. Machine learning algorithms learn patterns from training data, and if that data contains biases or reflects existing societal inequalities, the resulting AI systems may perpetuate or amplify these biases. This has been demonstrated in various domains, from facial recognition systems that perform poorly on certain demographic groups to hiring algorithms that discriminate against certain populations.
        
        The environmental impact of AI is also becoming an increasing concern. Training large neural networks requires significant computational resources, leading to substantial energy consumption and carbon emissions. Researchers are actively working on developing more energy-efficient algorithms and hardware, but the environmental costs of AI development remain a significant consideration.
        
        Despite these challenges, the potential benefits of AI are enormous and far-reaching. In healthcare, AI systems are being developed to assist in medical diagnosis, drug discovery, and personalized treatment planning. These systems can analyze medical images, identify patterns in patient data, and provide recommendations that complement human expertise, potentially improving outcomes and reducing costs.
        
        In education, AI-powered systems can provide personalized learning experiences, adapting to individual student needs and learning styles. These systems can identify areas where students struggle, provide targeted support, and track progress over time, enabling more effective and engaging educational experiences.
        
        Transportation is another domain where AI is having a profound impact. Autonomous vehicles, powered by sophisticated AI systems, have the potential to revolutionize how we move people and goods. These systems can improve safety by reducing human error, increase efficiency by optimizing routes and reducing congestion, and provide mobility options for people who cannot drive.
        
        The economic implications of AI are equally significant. AI has the potential to automate many routine and repetitive tasks, potentially leading to significant productivity gains and economic growth. However, this automation also raises concerns about job displacement and the need for workers to develop new skills and adapt to changing labor markets.
        
        The development of AI also raises important questions about the future of work and human-AI collaboration. Rather than replacing humans entirely, many AI systems are designed to augment human capabilities, working alongside people to solve complex problems and make better decisions. This collaborative approach, often called "human-in-the-loop" AI, combines the strengths of both human and artificial intelligence.
        
        As AI technology continues to advance, it is essential to consider the ethical implications and ensure that these systems are developed and deployed responsibly. This includes addressing issues of privacy, security, accountability, and the potential for misuse or unintended consequences. It also requires ongoing dialogue between technologists, policymakers, ethicists, and the public to ensure that AI development aligns with human values and societal goals.
        
        The future of AI is likely to be characterized by continued rapid advancement, with new capabilities and applications emerging regularly. Areas of active research include more sophisticated language models, improved reasoning and planning capabilities, and the development of AI systems that can learn from fewer examples and generalize more effectively across different domains.
        
        However, realizing the full potential of AI will require addressing the current challenges and limitations. This includes developing more interpretable and explainable AI systems, ensuring fairness and reducing bias, addressing environmental concerns, and establishing appropriate governance and regulatory frameworks.
        
        The development of AI also requires ongoing investment in research and development, education and training, and infrastructure. This includes not only technical research but also interdisciplinary work that brings together experts from computer science, psychology, philosophy, economics, and other fields to address the complex challenges and opportunities presented by AI.
        
        In conclusion, artificial intelligence represents one of the most important and transformative technologies of our time. While it presents significant challenges and requires careful consideration of ethical and societal implications, it also offers enormous potential for improving human lives and addressing some of the world's most pressing problems. The key to realizing this potential lies in responsible development, thoughtful deployment, and ongoing engagement with the broader implications of AI technology.
        """,
        
        "structured_text": """
        # Introduction to Machine Learning
        
        ## What is Machine Learning?
        Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.
        
        ## Types of Machine Learning
        
        ### Supervised Learning
        Supervised learning involves training a model on labeled data to make predictions on new, unseen data.
        
        ### Unsupervised Learning
        Unsupervised learning finds hidden patterns in data without predefined labels.
        
        ### Reinforcement Learning
        Reinforcement learning involves an agent learning to make decisions by taking actions in an environment.
        
        ## Applications
        
        ### Computer Vision
        - Image classification
        - Object detection
        - Facial recognition
        
        ### Natural Language Processing
        - Text classification
        - Machine translation
        - Sentiment analysis
        
        ### Healthcare
        - Medical diagnosis
        - Drug discovery
        - Patient monitoring
        
        ## Challenges and Limitations
        
        ### Data Quality
        - Insufficient data
        - Biased data
        - Noisy data
        
        ### Interpretability
        - Black box models
        - Lack of transparency
        - Difficulty in debugging
        
        ### Ethical Concerns
        - Privacy issues
        - Bias and discrimination
        - Job displacement
        
        ## Future Directions
        
        ### Explainable AI
        - Model interpretability
        - Decision explanations
        - Trust and accountability
        
        ### Federated Learning
        - Privacy-preserving training
        - Distributed learning
        - Collaborative models
        
        ### Quantum Machine Learning
        - Quantum algorithms
        - Quantum advantage
        - Hybrid approaches
        """
    }
    
    return data_sets

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return None

def benchmark_processing_speed(processor, data_sets):
    """Benchmark processing speed for different data sizes"""
    print("🚀 Processing Speed Benchmark")
    print("=" * 35)
    
    # Test different chunking configurations
    chunking_configs = [
        ("Small chunks", ChunkingOptions(max_chunk_size=200, overlap=25)),
        ("Medium chunks", ChunkingOptions(max_chunk_size=400, overlap=50)),
        ("Large chunks", ChunkingOptions(max_chunk_size=800, overlap=100)),
        ("Semantic chunks", ChunkingOptions(max_chunk_size=500, overlap=75, semantic_chunking=True))
    ]
    
    results = {}
    
    for config_name, options in chunking_configs:
        print(f"\n   🔧 Configuration: {config_name}")
        print(f"      Options: size={options.max_chunk_size}, overlap={options.overlap}, semantic={options.semantic_chunking}")
        
        config_results = {}
        
        for data_name, content in data_sets.items():
            print(f"      📝 Testing: {data_name} ({len(content)} chars)")
            
            # Run multiple iterations for more accurate timing
            times = []
            chunk_counts = []
            
            for iteration in range(3):  # 3 iterations for averaging
                gc.collect()  # Force garbage collection
                start_time = time.time()
                start_memory = get_memory_usage()
                
                try:
                    response = processor.process_transcript(content, chunking_options=options)
                    processing_time = time.time() - start_time
                    end_memory = get_memory_usage()
                    
                    times.append(processing_time)
                    chunk_counts.append(len(response.chunks))
                    
                    # Memory delta
                    memory_delta = end_memory - start_memory if start_memory and end_memory else 0
                    
                except Exception as e:
                    print(f"         ❌ Iteration {iteration + 1} failed: {e}")
                    continue
            
            if times:
                avg_time = statistics.mean(times)
                avg_chunks = statistics.mean(chunk_counts)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                
                print(f"         ✅ Avg time: {avg_time:.3f}s ± {std_time:.3f}s")
                print(f"         📊 Avg chunks: {avg_chunks:.1f}")
                print(f"         📏 Chars per second: {len(content) / avg_time:.0f}")
                
                config_results[data_name] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'avg_chunks': avg_chunks,
                    'chars_per_second': len(content) / avg_time
                }
            else:
                print(f"         ❌ All iterations failed")
                config_results[data_name] = None
        
        results[config_name] = config_results
    
    return results

def benchmark_memory_usage(processor, data_sets):
    """Benchmark memory usage patterns"""
    print("\n💾 Memory Usage Benchmark")
    print("=" * 30)
    
    # Test with large text to see memory patterns
    large_text = data_sets["large_text"]
    
    print("   📝 Testing memory usage with large text...")
    
    # Baseline memory
    gc.collect()
    baseline_memory = get_memory_usage()
    print(f"      📊 Baseline memory: {baseline_memory:.1f} MB")
    
    # Process with different chunk sizes
    chunk_sizes = [200, 400, 600, 800, 1000]
    memory_results = {}
    
    for size in chunk_sizes:
        options = ChunkingOptions(max_chunk_size=size, overlap=size//4)
        
        gc.collect()
        start_memory = get_memory_usage()
        
        try:
            response = processor.process_transcript(large_text, chunking_options=options)
            end_memory = get_memory_usage()
            
            memory_delta = end_memory - start_memory if start_memory and end_memory else 0
            chunks_created = len(response.chunks)
            
            print(f"      📏 Chunk size {size}: {memory_delta:.1f} MB delta, {chunks_created} chunks")
            
            memory_results[size] = {
                'memory_delta': memory_delta,
                'chunks_created': chunks_created,
                'memory_per_chunk': memory_delta / chunks_created if chunks_created > 0 else 0
            }
            
        except Exception as e:
            print(f"      ❌ Chunk size {size} failed: {e}")
            memory_results[size] = None
    
    return memory_results

def benchmark_query_performance(processor, data_sets):
    """Benchmark query performance"""
    print("\n🔍 Query Performance Benchmark")
    print("=" * 35)
    
    # First, process a medium text for querying
    medium_text = data_sets["medium_text"]
    print("   📝 Processing medium text for querying...")
    
    try:
        response = processor.process_transcript(
            medium_text,
            chunking_options=ChunkingOptions(max_chunk_size=400, overlap=50)
        )
        print(f"      ✅ Created {len(response.chunks)} chunks for querying")
    except Exception as e:
        print(f"      ❌ Processing failed: {e}")
        return {}
    
    # Test different query types
    query_types = [
        ("Simple fact", "What is machine learning?"),
        ("Definition", "Define deep learning"),
        ("Comparison", "How do supervised and unsupervised learning differ?"),
        ("Application", "What are the applications of machine learning?"),
        ("Complex", "What are the challenges and limitations of machine learning in healthcare?")
    ]
    
    query_configs = [
        ("Basic", QueryOptions()),
        ("Show chunks", QueryOptions(show_chunks=True, max_chunks=2)),
        ("Show Q&A", QueryOptions(show_prompt_answers=True, max_chunks=3)),
        ("Comprehensive", QueryOptions(show_chunks=True, show_prompt_answers=True, max_chunks=5))
    ]
    
    query_results = {}
    
    for query_name, query_text in query_types:
        print(f"\n   🤔 Query: {query_name}")
        query_results[query_name] = {}
        
        for config_name, options in query_configs:
            print(f"      🔧 {config_name}:")
            
            # Run multiple iterations
            times = []
            for iteration in range(3):
                start_time = time.time()
                try:
                    result = processor.query(query_text, query_options=options)
                    query_time = time.time() - start_time
                    times.append(query_time)
                except Exception as e:
                    print(f"         ❌ Iteration {iteration + 1} failed: {e}")
                    continue
            
            if times:
                avg_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                
                print(f"         ⏱️  Avg time: {avg_time:.3f}s ± {std_time:.3f}s")
                
                query_results[query_name][config_name] = {
                    'avg_time': avg_time,
                    'std_time': std_time
                }
            else:
                print(f"         ❌ All iterations failed")
                query_results[query_name][config_name] = None
    
    return query_results

def benchmark_scalability(processor):
    """Benchmark scalability with increasing data sizes"""
    print("\n📈 Scalability Benchmark")
    print("=" * 30)
    
    # Generate progressively larger texts
    base_text = "This is a sample text about artificial intelligence and machine learning. "
    text_sizes = [1000, 5000, 10000, 25000, 50000]  # characters
    
    scalability_results = {}
    
    for size in text_sizes:
        # Generate text of specified size
        repetitions = size // len(base_text) + 1
        test_text = (base_text * repetitions)[:size]
        
        print(f"   📝 Testing text size: {size:,} characters")
        
        # Test with medium chunk size
        options = ChunkingOptions(max_chunk_size=400, overlap=50)
        
        gc.collect()
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            response = processor.process_transcript(test_text, chunking_options=options)
            processing_time = time.time() - start_time
            end_memory = get_memory_usage()
            
            chunks_created = len(response.chunks)
            memory_delta = end_memory - start_memory if start_memory and end_memory else 0
            
            print(f"      ✅ Time: {processing_time:.3f}s, Chunks: {chunks_created}, Memory: {memory_delta:.1f} MB")
            print(f"      📊 Throughput: {size / processing_time:.0f} chars/sec")
            
            scalability_results[size] = {
                'processing_time': processing_time,
                'chunks_created': chunks_created,
                'memory_delta': memory_delta,
                'throughput': size / processing_time
            }
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            scalability_results[size] = None
    
    return scalability_results

def analyze_benchmark_results(processing_results, memory_results, query_results, scalability_results):
    """Analyze and summarize benchmark results"""
    print("\n📊 Benchmark Results Analysis")
    print("=" * 35)
    
    # Processing speed analysis
    print("\n   🚀 Processing Speed Analysis:")
    if processing_results:
        best_config = None
        best_performance = 0
        
        for config_name, config_results in processing_results.items():
            if not config_results:
                continue
                
            # Calculate average performance across all data sizes
            performances = []
            for data_name, result in config_results.items():
                if result and result.get('chars_per_second'):
                    performances.append(result['chars_per_second'])
            
            if performances:
                avg_performance = statistics.mean(performances)
                print(f"      {config_name}: {avg_performance:.0f} chars/sec average")
                
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_config = config_name
        
        if best_config:
            print(f"      🏆 Best configuration: {best_config}")
    
    # Memory efficiency analysis
    print("\n   💾 Memory Efficiency Analysis:")
    if memory_results:
        memory_per_chunk = []
        for size, result in memory_results.items():
            if result and result.get('memory_per_chunk'):
                memory_per_chunk.append((size, result['memory_per_chunk']))
        
        if memory_per_chunk:
            # Find most memory-efficient chunk size
            most_efficient = min(memory_per_chunk, key=lambda x: x[1])
            print(f"      🏆 Most memory-efficient: {most_efficient[0]} chars per chunk")
            print(f"         Memory per chunk: {most_efficient[1]:.3f} MB")
    
    # Query performance analysis
    print("\n   🔍 Query Performance Analysis:")
    if query_results:
        query_times = []
        for query_name, configs in query_results.items():
            for config_name, result in configs.items():
                if result and result.get('avg_time'):
                    query_times.append((query_name, config_name, result['avg_time']))
        
        if query_times:
            # Find fastest queries
            fastest_queries = sorted(query_times, key=lambda x: x[2])[:3]
            print(f"      🏆 Top 3 fastest queries:")
            for i, (query, config, time) in enumerate(fastest_queries, 1):
                print(f"         {i}. {query} ({config}): {time:.3f}s")
    
    # Scalability analysis
    print("\n   📈 Scalability Analysis:")
    if scalability_results:
        throughputs = []
        for size, result in scalability_results.items():
            if result and result.get('throughput'):
                throughputs.append((size, result['throughput']))
        
        if throughputs:
            # Check if throughput scales linearly
            throughputs.sort(key=lambda x: x[0])
            if len(throughputs) >= 2:
                first_throughput = throughputs[0][1]
                last_throughput = throughputs[-1][1]
                scaling_factor = last_throughput / first_throughput
                size_factor = throughputs[-1][0] / throughputs[0][0]
                
                print(f"      📊 Throughput scaling: {scaling_factor:.2f}x for {size_factor:.1f}x size increase")
                
                if scaling_factor > size_factor * 0.8:
                    print("      ✅ Good scalability (near-linear)")
                elif scaling_factor > size_factor * 0.5:
                    print("      ⚠️  Moderate scalability")
                else:
                    print("      ❌ Poor scalability")

def generate_optimization_recommendations(processing_results, memory_results, query_results, scalability_results):
    """Generate optimization recommendations based on benchmark results"""
    print("\n💡 Optimization Recommendations")
    print("=" * 35)
    
    recommendations = []
    
    # Processing speed recommendations
    if processing_results:
        print("\n   🚀 Processing Speed:")
        
        # Find best performing configuration
        best_config = None
        best_performance = 0
        for config_name, config_results in processing_results.items():
            if not config_results:
                continue
            performances = [r['chars_per_second'] for r in config_results.values() if r and r.get('chars_per_second')]
            if performances:
                avg_performance = statistics.mean(performances)
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_config = config_name
        
        if best_config:
            print(f"      ✅ Use {best_config} for best processing speed")
            recommendations.append(f"Use {best_config} configuration for optimal processing speed")
        
        # Check for performance variations
        for config_name, config_results in processing_results.items():
            if not config_results:
                continue
            times = [r['avg_time'] for r in config_results.values() if r and r.get('avg_time')]
            if times and len(times) > 1:
                cv = statistics.stdev(times) / statistics.mean(times)
                if cv > 0.2:
                    print(f"      ⚠️  {config_name} shows high variability (CV: {cv:.2f})")
                    recommendations.append(f"Investigate performance variability in {config_name} configuration")
    
    # Memory recommendations
    if memory_results:
        print("\n   💾 Memory Usage:")
        
        # Find most memory-efficient chunk size
        memory_per_chunk = [(size, result['memory_per_chunk']) for size, result in memory_results.items() 
                           if result and result.get('memory_per_chunk')]
        if memory_per_chunk:
            most_efficient = min(memory_per_chunk, key=lambda x: x[1])
            print(f"      ✅ Most memory-efficient: {most_efficient[0]} chars per chunk")
            recommendations.append(f"Use {most_efficient[0]} character chunks for optimal memory efficiency")
    
    # Query performance recommendations
    if query_results:
        print("\n   🔍 Query Performance:")
        
        # Identify slow queries
        slow_queries = []
        for query_name, configs in query_results.items():
            for config_name, result in configs.items():
                if result and result.get('avg_time') and result['avg_time'] > 2.0:  # > 2 seconds
                    slow_queries.append((query_name, config_name, result['avg_time']))
        
        if slow_queries:
            print(f"      ⚠️  {len(slow_queries)} slow queries detected:")
            for query, config, time in slow_queries[:3]:  # Show top 3
                print(f"         {query} ({config}): {time:.2f}s")
            recommendations.append("Investigate slow queries and consider query optimization")
    
    # Scalability recommendations
    if scalability_results:
        print("\n   📈 Scalability:")
        
        # Check throughput scaling
        throughputs = [(size, result['throughput']) for size, result in scalability_results.items() 
                      if result and result.get('throughput')]
        if len(throughputs) >= 2:
            throughputs.sort(key=lambda x: x[0])
            scaling_factor = throughputs[-1][1] / throughputs[0][1]
            size_factor = throughputs[-1][0] / throughputs[0][0]
            
            if scaling_factor < size_factor * 0.5:
                print(f"      ❌ Poor scalability detected")
                recommendations.append("Investigate scalability bottlenecks for large documents")
            elif scaling_factor < size_factor * 0.8:
                print(f"      ⚠️  Moderate scalability")
                recommendations.append("Consider optimizations for better scalability")
            else:
                print(f"      ✅ Good scalability maintained")
    
    # General recommendations
    print("\n   🔧 General Recommendations:")
    
    if not recommendations:
        print("      ✅ All benchmarks show good performance")
        recommendations.append("Current configuration is well-optimized")
    
    for i, recommendation in enumerate(recommendations, 1):
        print(f"         {i}. {recommendation}")
    
    return recommendations

def export_benchmark_results(processing_results, memory_results, query_results, scalability_results, recommendations):
    """Export benchmark results to JSON file"""
    print("\n📤 Exporting Benchmark Results")
    print("=" * 35)
    
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'processing_configs_tested': len(processing_results),
            'memory_tests_performed': len(memory_results),
            'query_types_tested': len(query_results),
            'scalability_levels_tested': len(scalability_results)
        },
        'processing_results': processing_results,
        'memory_results': memory_results,
        'query_results': query_results,
        'scalability_results': scalability_results,
        'recommendations': recommendations
    }
    
    filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"   ✅ Results exported to: {filename}")
        return filename
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        return None

def main():
    """Main performance benchmarking function"""
    print("🎯 EchoGem Performance Benchmarking Demonstration")
    print("=" * 70)
    print("This demo will comprehensively test EchoGem's performance across various scenarios!")
    print()
    
    # Initialize processor
    print("1️⃣ Initializing EchoGem Processor...")
    try:
        processor = Processor()
        print("   ✅ Processor initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize processor: {e}")
        print("   💡 Make sure your API keys are set correctly")
        return
    
    # Create benchmark data
    print("\n2️⃣ Creating benchmark data sets...")
    data_sets = create_benchmark_data()
    for name, content in data_sets.items():
        print(f"   📝 {name}: {len(content):,} characters")
    
    # Run benchmarks
    print("\n3️⃣ Running Performance Benchmarks...")
    processing_results = benchmark_processing_speed(processor, data_sets)
    memory_results = benchmark_memory_usage(processor, data_sets)
    query_results = benchmark_query_performance(processor, data_sets)
    scalability_results = benchmark_scalability(processor)
    
    # Analyze results
    print("\n4️⃣ Analyzing Benchmark Results...")
    analyze_benchmark_results(processing_results, memory_results, query_results, scalability_results)
    
    # Generate recommendations
    print("\n5️⃣ Generating Optimization Recommendations...")
    recommendations = generate_optimization_recommendations(
        processing_results, memory_results, query_results, scalability_results
    )
    
    # Export results
    print("\n6️⃣ Exporting Results...")
    export_file = export_benchmark_results(
        processing_results, memory_results, query_results, scalability_results, recommendations
    )
    
    # Final summary
    print("\n🎉 Performance Benchmarking Complete!")
    print("=" * 40)
    print("💡 Key findings:")
    print(f"   📊 Processing configurations tested: {len(processing_results)}")
    print(f"   💾 Memory efficiency analyzed: {len(memory_results)} scenarios")
    print(f"   🔍 Query performance measured: {len(query_results)} query types")
    print(f"   📈 Scalability tested: {len(scalability_results)} document sizes")
    print(f"   💡 Recommendations generated: {len(recommendations)}")
    
    if export_file:
        print(f"\n📁 Detailed results exported to: {export_file}")
    
    print("\n📚 Explore other demos:")
    print("   - Basic workflow: python demos/01_basic_workflow_demo.py")
    print("   - CLI usage: python demos/02_cli_demo.py")
    print("   - Python API: python demos/03_api_demo.py")
    print("   - Academic papers: python demos/04_academic_paper_demo.py")

if __name__ == "__main__":
    main()
