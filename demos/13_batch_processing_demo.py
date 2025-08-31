#!/usr/bin/env python3
"""
EchoGem Batch Processing Demo

This demo showcases large-scale document processing:
- Batch processing multiple documents
- Progress tracking and monitoring
- Error handling and recovery
- Performance optimization
- Resource management
"""

import os
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import queue

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from processor import Processor
from models import ChunkingOptions, QueryOptions

def create_batch_documents():
    """Create multiple sample documents for batch processing"""
    print("📝 Creating batch processing documents...")
    
    documents = {}
    
    # Create documents of varying sizes and types
    for i in range(1, 21):  # 20 documents
        if i <= 5:
            # Small documents (technical concepts)
            content = f"""
            Technical Concept {i}: Advanced Machine Learning
            
            This document covers advanced machine learning concepts including neural networks, deep learning architectures, and optimization techniques. Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.
            
            Key topics include supervised learning, unsupervised learning, and reinforcement learning. Each approach has its own strengths and applications in solving real-world problems.
            
            Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information and learn patterns from training data.
            
            Deep learning extends neural networks with multiple layers to create hierarchical representations of data. This approach has revolutionized fields like computer vision and natural language processing.
            
            Optimization techniques are crucial for training effective models. Gradient descent, stochastic gradient descent, and adaptive optimization methods help models converge to optimal solutions.
            """
            doc_type = "technical_small"
        elif i <= 10:
            # Medium documents (research summaries)
            content = f"""
            Research Summary {i}: AI Applications in Healthcare
            
            Artificial intelligence is transforming healthcare through various applications including medical diagnosis, drug discovery, patient monitoring, and personalized treatment planning. These technologies have the potential to improve patient outcomes while reducing healthcare costs.
            
            Medical imaging analysis using AI has shown remarkable accuracy in detecting diseases like cancer, heart disease, and neurological disorders. Deep learning models can analyze X-rays, CT scans, and MRI images to identify abnormalities that human radiologists might miss.
            
            Drug discovery is another area where AI is making significant contributions. Machine learning algorithms can predict drug efficacy, identify potential side effects, and accelerate the development of new treatments. This has the potential to reduce the time and cost of bringing new drugs to market.
            
            Patient monitoring systems powered by AI can track vital signs, detect early warning signs of complications, and alert healthcare providers to potential issues. This enables proactive care and can prevent adverse events.
            
            Personalized medicine uses AI to analyze patient data and develop customized treatment plans. By considering genetic factors, medical history, and lifestyle factors, AI can recommend treatments that are most likely to be effective for individual patients.
            
            Despite the promise of AI in healthcare, there are challenges including data privacy, regulatory compliance, and the need for clinical validation. Ensuring the safety and effectiveness of AI-powered healthcare tools is crucial for widespread adoption.
            """
            doc_type = "research_medium"
        elif i <= 15:
            # Large documents (comprehensive guides)
            content = f"""
            Comprehensive Guide {i}: Complete Data Science Workflow
            
            Data science is an interdisciplinary field that combines statistics, machine learning, data analysis, and domain expertise to extract insights from data. This comprehensive guide covers the entire data science workflow from data collection to model deployment.
            
            The first stage of the data science workflow is data collection. This involves gathering data from various sources including databases, APIs, web scraping, sensors, and surveys. The quality and quantity of data significantly impact the success of data science projects. It's important to ensure data is relevant, accurate, and comprehensive.
            
            Data cleaning and preprocessing are critical steps that involve handling missing values, removing duplicates, dealing with outliers, and ensuring data consistency. Missing values can be handled through imputation techniques, deletion, or advanced methods like multiple imputation. Outliers should be identified and either removed or handled appropriately based on the analysis goals.
            
            Exploratory data analysis (EDA) involves examining and visualizing data to understand patterns, relationships, and anomalies. This stage helps identify potential issues in the data and guides the modeling approach. EDA includes descriptive statistics, data visualization, and correlation analysis. Visualization tools like histograms, scatter plots, and heatmaps help identify patterns and relationships in the data.
            
            Feature engineering involves creating new features from existing data that can improve model performance. This includes feature selection, transformation, and creation of interaction terms. Feature selection methods help identify the most relevant variables for modeling, while feature transformation techniques like normalization and standardization ensure features are on similar scales.
            
            Model development involves selecting appropriate algorithms, tuning hyperparameters, and training models. The choice of algorithm depends on the problem type (classification, regression, clustering), data characteristics, and performance requirements. Common algorithms include linear regression, logistic regression, decision trees, random forests, support vector machines, and neural networks.
            
            Model evaluation involves assessing model performance using appropriate metrics and validation techniques. For classification problems, metrics like accuracy, precision, recall, F1-score, and ROC-AUC are commonly used. For regression problems, metrics like mean squared error, mean absolute error, and R-squared are appropriate. Cross-validation techniques help ensure models generalize well to new data.
            
            Model deployment involves integrating trained models into production systems where they can make predictions on new data. This requires careful consideration of scalability, reliability, and monitoring. Models should be regularly retrained with new data to maintain performance over time.
            
            The data science workflow is iterative, with insights from each stage informing subsequent stages. Continuous learning and adaptation are key to successful data science projects. Collaboration between data scientists, domain experts, and stakeholders is essential for ensuring that insights are actionable and valuable.
            """
            doc_type = "guide_large"
        else:
            # Extra large documents (detailed reports)
            content = f"""
            Detailed Report {i}: Enterprise AI Implementation Strategy
            
            Implementing artificial intelligence in enterprise environments requires careful planning, strategic vision, and systematic execution. This detailed report provides a comprehensive framework for successful AI implementation across various business functions and industries.
            
            The first step in enterprise AI implementation is establishing a clear strategic vision. Organizations must identify specific business problems that AI can solve and define measurable success criteria. This involves conducting a thorough assessment of current business processes, identifying pain points, and evaluating the potential impact of AI solutions. Stakeholder buy-in is crucial at this stage, as AI implementation often requires significant organizational change and investment.
            
            Technology infrastructure assessment is the next critical step. Organizations must evaluate their current data infrastructure, computational resources, and technical capabilities. This includes assessing data quality, availability, and accessibility. Many organizations find that their existing infrastructure needs significant upgrades to support AI workloads. Cloud-based solutions often provide the flexibility and scalability needed for AI implementation, but on-premises solutions may be preferred for security or compliance reasons.
            
            Data strategy development is essential for AI success. Organizations must establish clear policies for data collection, storage, processing, and governance. This includes defining data quality standards, establishing data lineage tracking, and implementing appropriate security and privacy measures. Data strategy should also address issues of data bias, fairness, and ethical considerations. Organizations must ensure that their AI systems treat all stakeholders fairly and do not perpetuate existing biases or discrimination.
            
            Team building and skill development are crucial for AI implementation success. Organizations need to hire or develop talent with expertise in machine learning, data engineering, and AI ethics. This may involve creating new roles, providing training for existing employees, or partnering with external consultants. Cross-functional teams that include business stakeholders, technical experts, and domain specialists are most effective for AI implementation projects.
            
            Pilot project selection and execution help organizations learn and refine their AI implementation approach. Starting with smaller, well-defined projects allows organizations to build confidence, develop best practices, and demonstrate value before scaling up. Pilot projects should be chosen based on their potential for quick wins, clear success metrics, and manageable scope. Regular review and iteration of pilot projects help organizations learn and improve their approach.
            
            Change management and organizational adoption are critical for AI implementation success. AI often requires changes to business processes, job roles, and organizational culture. Organizations must communicate the benefits of AI clearly, address concerns about job displacement, and provide training and support for employees affected by AI implementation. Change management should be proactive and ongoing throughout the implementation process.
            
            Performance monitoring and continuous improvement ensure that AI systems deliver ongoing value. Organizations must establish metrics for measuring AI performance, implement monitoring systems to track system behavior, and establish processes for regular review and improvement. This includes monitoring for model drift, performance degradation, and unexpected behavior. Regular retraining and updating of AI models helps maintain performance over time.
            
            Risk management and governance are essential for responsible AI implementation. Organizations must identify and mitigate risks associated with AI systems, including technical risks, business risks, and ethical risks. This includes implementing appropriate controls, establishing governance frameworks, and ensuring compliance with relevant regulations and standards. Regular risk assessments and audits help organizations identify and address emerging risks.
            
            Scaling and expansion involve extending successful AI implementations across the organization. This requires standardizing processes, developing reusable components, and establishing centers of excellence for AI development and deployment. Organizations should focus on building scalable infrastructure and processes that can support multiple AI applications and use cases.
            
            The future of enterprise AI involves continued evolution and innovation. Organizations must stay current with emerging AI technologies, evaluate new opportunities for AI application, and continuously refine their AI strategy and implementation approach. This requires ongoing investment in research and development, partnerships with technology providers and research institutions, and participation in industry forums and standards development.
            
            Successful enterprise AI implementation requires a holistic approach that addresses technical, organizational, and cultural factors. Organizations that take a systematic approach to AI implementation, focus on business value, and invest in the necessary infrastructure and talent are most likely to achieve success. The journey to AI maturity is ongoing, requiring continuous learning, adaptation, and improvement.
            """
            doc_type = "report_xlarge"
        
        documents[f"document_{i:02d}_{doc_type}"] = {
            'content': content.strip(),
            'type': doc_type,
            'size': len(content)
        }
    
    print(f"   ✅ Created {len(documents)} documents for batch processing")
    for doc_type in set(doc['type'] for doc in documents.values()):
        count = len([d for d in documents.values() if d['type'] == doc_type])
        total_size = sum(d['size'] for d in documents.values() if d['type'] == doc_type)
        print(f"      {doc_type}: {count} documents, {total_size:,} total characters")
    
    return documents

def process_single_document(processor, doc_name, doc_data, chunking_options, progress_queue):
    """Process a single document and return results"""
    try:
        start_time = time.time()
        
        # Process the document
        response = processor.process_transcript(doc_data['content'], chunking_options=chunking_options)
        
        processing_time = time.time() - start_time
        
        result = {
            'doc_name': doc_name,
            'doc_type': doc_data['type'],
            'doc_size': doc_data['size'],
            'chunks_created': len(response.chunks),
            'processing_time': processing_time,
            'chars_per_second': doc_data['size'] / processing_time if processing_time > 0 else 0,
            'success': True,
            'error': None
        }
        
        # Update progress
        progress_queue.put(('success', doc_name, result))
        
        return result
        
    except Exception as e:
        error_result = {
            'doc_name': doc_name,
            'doc_type': doc_data['type'],
            'doc_size': doc_data['size'],
            'chunks_created': 0,
            'processing_time': 0,
            'chars_per_second': 0,
            'success': False,
            'error': str(e)
        }
        
        # Update progress
        progress_queue.put(('error', doc_name, error_result))
        
        return error_result

def demo_sequential_processing(processor, documents):
    """Demonstrate sequential document processing"""
    print("\n📝 Sequential Processing Demo")
    print("=" * 35)
    
    # Configure chunking options
    chunking_options = ChunkingOptions(
        max_chunk_size=500,
        overlap=75,
        semantic_chunking=True
    )
    
    print("   🔧 Processing documents sequentially...")
    print(f"      Total documents: {len(documents)}")
    print(f"      Chunking options: size={chunking_options.max_chunk_size}, overlap={chunking_options.overlap}")
    
    start_time = time.time()
    results = []
    
    for i, (doc_name, doc_data) in enumerate(documents.items(), 1):
        print(f"      📄 Processing {i}/{len(documents)}: {doc_name}")
        
        try:
            result = process_single_document(processor, doc_name, doc_data, chunking_options, queue.Queue())
            results.append(result)
            
            if result['success']:
                print(f"         ✅ {result['chunks_created']} chunks in {result['processing_time']:.2f}s")
            else:
                print(f"         ❌ Failed: {result['error']}")
                
        except Exception as e:
            print(f"         ❌ Processing error: {e}")
            results.append({
                'doc_name': doc_name,
                'doc_type': doc_data['type'],
                'doc_size': doc_data['size'],
                'chunks_created': 0,
                'processing_time': 0,
                'chars_per_second': 0,
                'success': False,
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n   📊 Sequential Processing Results:")
    print(f"      Total time: {total_time:.2f}s")
    print(f"      Successful: {len(successful_results)}")
    print(f"      Failed: {len(failed_results)}")
    
    if successful_results:
        total_chunks = sum(r['chunks_created'] for r in successful_results)
        total_chars = sum(r['doc_size'] for r in successful_results)
        avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        avg_chars_per_second = sum(r['chars_per_second'] for r in successful_results) / len(successful_results)
        
        print(f"      Total chunks created: {total_chunks}")
        print(f"      Total characters processed: {total_chars:,}")
        print(f"      Average processing time: {avg_processing_time:.2f}s")
        print(f"      Average throughput: {avg_chars_per_second:.0f} chars/sec")
    
    return results, total_time

def demo_parallel_processing(processor, documents, max_workers=4):
    """Demonstrate parallel document processing"""
    print(f"\n🚀 Parallel Processing Demo (max_workers={max_workers})")
    print("=" * 50)
    
    # Configure chunking options
    chunking_options = ChunkingOptions(
        max_chunk_size=500,
        overlap=75,
        semantic_chunking=True
    )
    
    print("   🔧 Processing documents in parallel...")
    print(f"      Total documents: {len(documents)}")
    print(f"      Max workers: {max_workers}")
    print(f"      Chunking options: size={chunking_options.max_chunk_size}, overlap={chunking_options.overlap}")
    
    start_time = time.time()
    results = []
    progress_queue = queue.Queue()
    
    # Progress monitoring thread
    def monitor_progress():
        completed = 0
        total = len(documents)
        while completed < total:
            try:
                status, doc_name, result = progress_queue.get(timeout=1)
                completed += 1
                if status == 'success':
                    print(f"         ✅ {completed}/{total}: {doc_name} - {result['chunks_created']} chunks")
                else:
                    print(f"         ❌ {completed}/{total}: {doc_name} - {result['error']}")
            except queue.Empty:
                continue
    
    # Start progress monitoring
    progress_thread = threading.Thread(target=monitor_progress)
    progress_thread.start()
    
    # Process documents in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_doc = {
            executor.submit(process_single_document, processor, doc_name, doc_data, chunking_options, progress_queue): doc_name
            for doc_name, doc_data in documents.items()
        }
        
        # Collect results
        for future in as_completed(future_to_doc):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                doc_name = future_to_doc[future]
                print(f"         ❌ {doc_name} - Future execution error: {e}")
                results.append({
                    'doc_name': doc_name,
                    'doc_type': 'unknown',
                    'doc_size': 0,
                    'chunks_created': 0,
                    'processing_time': 0,
                    'chars_per_second': 0,
                    'success': False,
                    'error': str(e)
                })
    
    # Wait for progress monitoring to complete
    progress_thread.join()
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n   📊 Parallel Processing Results:")
    print(f"      Total time: {total_time:.2f}s")
    print(f"      Successful: {len(successful_results)}")
    print(f"      Failed: {len(failed_results)}")
    
    if successful_results:
        total_chunks = sum(r['chunks_created'] for r in successful_results)
        total_chars = sum(r['doc_size'] for r in successful_results)
        avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        avg_chars_per_second = sum(r['chars_per_second'] for r in successful_results) / len(successful_results)
        
        print(f"      Total chunks created: {total_chunks}")
        print(f"      Total characters processed: {total_chars:,}")
        print(f"      Average processing time: {avg_processing_time:.2f}s")
        print(f"      Average throughput: {avg_chars_per_second:.0f} chars/sec")
    
    return results, total_time

def demo_batch_optimization(processor, documents):
    """Demonstrate batch processing optimization"""
    print("\n⚡ Batch Processing Optimization Demo")
    print("=" * 40)
    
    # Test different chunking configurations
    chunking_configs = [
        ("Small chunks", ChunkingOptions(max_chunk_size=300, overlap=50)),
        ("Medium chunks", ChunkingOptions(max_chunk_size=500, overlap=75)),
        ("Large chunks", ChunkingOptions(max_chunk_size=800, overlap=100)),
        ("Semantic chunks", ChunkingOptions(max_chunk_size=500, overlap=75, semantic_chunking=True))
    ]
    
    # Test with a subset of documents for optimization
    test_docs = dict(list(documents.items())[:5])  # First 5 documents
    
    print("   🔧 Testing different chunking configurations...")
    print(f"      Test documents: {len(test_docs)}")
    
    optimization_results = {}
    
    for config_name, options in chunking_configs:
        print(f"\n      🔧 Testing: {config_name}")
        print(f"         Options: size={options.max_chunk_size}, overlap={options.overlap}, semantic={options.semantic_chunking}")
        
        start_time = time.time()
        results = []
        
        for doc_name, doc_data in test_docs.items():
            try:
                result = process_single_document(processor, doc_name, doc_data, options, queue.Queue())
                results.append(result)
            except Exception as e:
                results.append({
                    'doc_name': doc_name,
                    'success': False,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        successful_results = [r for r in results if r['success']]
        if successful_results:
            total_chunks = sum(r['chunks_created'] for r in successful_results)
            total_chars = sum(r['doc_size'] for r in successful_results)
            avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
            throughput = total_chars / total_time if total_time > 0 else 0
            
            print(f"         ✅ Time: {total_time:.2f}s, Chunks: {total_chunks}, Throughput: {throughput:.0f} chars/sec")
            
            optimization_results[config_name] = {
                'total_time': total_time,
                'total_chunks': total_chunks,
                'throughput': throughput,
                'avg_processing_time': avg_processing_time
            }
        else:
            print(f"         ❌ All documents failed")
            optimization_results[config_name] = None
    
    # Find optimal configuration
    valid_results = {k: v for k, v in optimization_results.items() if v is not None}
    if valid_results:
        print(f"\n      🏆 Optimization Results:")
        
        # Best throughput
        best_throughput = max(valid_results.items(), key=lambda x: x[1]['throughput'])
        print(f"         Best throughput: {best_throughput[0]} ({best_throughput[1]['throughput']:.0f} chars/sec)")
        
        # Fastest processing
        fastest = min(valid_results.items(), key=lambda x: x[1]['total_time'])
        print(f"         Fastest processing: {fastest[0]} ({fastest[1]['total_time']:.2f}s)")
        
        # Most chunks (potential for better search)
        most_chunks = max(valid_results.items(), key=lambda x: x[1]['total_chunks'])
        print(f"         Most chunks: {most_chunks[0]} ({most_chunks[1]['total_chunks']} chunks)")
    
    return optimization_results

def demo_error_handling_and_recovery(processor, documents):
    """Demonstrate error handling and recovery in batch processing"""
    print("\n🛡️  Error Handling and Recovery Demo")
    print("=" * 40)
    
    # Create some problematic documents
    problematic_docs = {
        "empty_document": {'content': '', 'type': 'empty', 'size': 0},
        "very_long_line": {'content': 'A' * 10000, 'type': 'long_line', 'size': 10000},
        "special_chars": {'content': '🚀🌟💻🎯📊🔍💡⚡🏆📝', 'type': 'special', 'size': 20},
        "normal_document": {'content': 'This is a normal document for testing error handling.', 'type': 'normal', 'size': 60}
    }
    
    print("   🔧 Testing error handling with problematic documents...")
    
    chunking_options = ChunkingOptions(max_chunk_size=500, overlap=50)
    results = []
    
    for doc_name, doc_data in problematic_docs.items():
        print(f"\n      📄 Testing: {doc_name}")
        
        try:
            result = process_single_document(processor, doc_name, doc_data, chunking_options, queue.Queue())
            results.append(result)
            
            if result['success']:
                print(f"         ✅ Processed successfully: {result['chunks_created']} chunks")
            else:
                print(f"         ❌ Failed as expected: {result['error']}")
                
        except Exception as e:
            print(f"         ❌ Exception caught: {e}")
            results.append({
                'doc_name': doc_name,
                'doc_type': doc_data['type'],
                'doc_size': doc_data['size'],
                'chunks_created': 0,
                'processing_time': 0,
                'chars_per_second': 0,
                'success': False,
                'error': str(e)
            })
    
    # Recovery strategies
    print(f"\n      🔄 Recovery Strategies:")
    print(f"         Empty documents: Skip or use default content")
    print(f"         Very long lines: Truncate or split")
    print(f"         Special characters: Encode or normalize")
    print(f"         Processing errors: Retry with different options")
    
    return results

def demo_resource_management(processor, documents):
    """Demonstrate resource management in batch processing"""
    print("\n💾 Resource Management Demo")
    print("=" * 35)
    
    print("   🔧 Testing resource management...")
    
    # Test memory usage with different batch sizes
    batch_sizes = [1, 5, 10, 20]
    
    chunking_options = ChunkingOptions(max_chunk_size=500, overlap=50)
    
    for batch_size in batch_sizes:
        print(f"\n      📦 Batch size: {batch_size}")
        
        # Process documents in batches
        doc_items = list(documents.items())
        total_batches = (len(doc_items) + batch_size - 1) // batch_size
        
        start_time = time.time()
        total_chunks = 0
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(doc_items))
            batch_docs = doc_items[batch_start:batch_end]
            
            batch_results = []
            for doc_name, doc_data in batch_docs:
                try:
                    result = process_single_document(processor, doc_name, doc_data, chunking_options, queue.Queue())
                    batch_results.append(result)
                    if result['success']:
                        total_chunks += result['chunks_created']
                except Exception as e:
                    print(f"            ❌ {doc_name}: {e}")
            
            print(f"         Batch {batch_num + 1}/{total_batches}: {len(batch_results)} documents")
        
        total_time = time.time() - start_time
        
        print(f"         Total time: {total_time:.2f}s")
        print(f"         Total chunks: {total_chunks}")
        print(f"         Throughput: {total_chunks / total_time:.1f} chunks/sec")
    
    # Memory management recommendations
    print(f"\n      💡 Resource Management Recommendations:")
    print(f"         Use appropriate batch sizes based on available memory")
    print(f"         Monitor memory usage during processing")
    print(f"         Implement garbage collection between batches")
    print(f"         Consider streaming for very large documents")
    print(f"         Use progress tracking to monitor resource consumption")

def export_batch_results(sequential_results, parallel_results, optimization_results, error_results):
    """Export batch processing results to JSON file"""
    print("\n📤 Exporting Batch Processing Results")
    print("=" * 40)
    
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'sequential_processing': {
                'total_documents': len(sequential_results),
                'successful': len([r for r in sequential_results if r['success']]),
                'failed': len([r for r in sequential_results if not r['success']])
            },
            'parallel_processing': {
                'total_documents': len(parallel_results),
                'successful': len([r for r in parallel_results if r['success']]),
                'failed': len([r for r in parallel_results if not r['success']])
            },
            'optimization_tests': len([r for r in optimization_results.values() if r is not None]),
            'error_handling_tests': len(error_results)
        },
        'sequential_results': sequential_results,
        'parallel_results': parallel_results,
        'optimization_results': optimization_results,
        'error_handling_results': error_results,
        'analysis_metadata': {
            'analysis_date': datetime.now().isoformat(),
            'analysis_type': 'comprehensive_batch_processing'
        }
    }
    
    filename = f"batch_processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"   ✅ Results exported to: {filename}")
        
        # Show export summary
        print(f"   📊 Export Summary:")
        print(f"      Sequential processing: {export_data['summary']['sequential_processing']['total_documents']} documents")
        print(f"      Parallel processing: {export_data['summary']['parallel_processing']['total_documents']} documents")
        print(f"      Optimization tests: {export_data['summary']['optimization_tests']}")
        print(f"      Error handling tests: {export_data['summary']['error_handling_tests']}")
        
        return filename
        
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        return None

def main():
    """Main batch processing demo function"""
    print("🎯 EchoGem Batch Processing Demonstration")
    print("=" * 70)
    print("This demo showcases large-scale document processing capabilities!")
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
    
    # Create batch documents
    print("\n2️⃣ Creating batch processing documents...")
    documents = create_batch_documents()
    
    # Run batch processing demos
    print("\n3️⃣ Running Batch Processing Demos...")
    
    # Sequential processing
    sequential_results, sequential_time = demo_sequential_processing(processor, documents)
    
    # Parallel processing
    parallel_results, parallel_time = demo_parallel_processing(processor, documents, max_workers=4)
    
    # Performance comparison
    if sequential_time > 0 and parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"\n📊 Performance Comparison:")
        print(f"   Sequential time: {sequential_time:.2f}s")
        print(f"   Parallel time: {parallel_time:.2f}s")
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 1:
            print(f"   ✅ Parallel processing is {speedup:.2f}x faster")
        else:
            print(f"   ⚠️  Sequential processing is faster (possible overhead)")
    
    # Optimization demo
    optimization_results = demo_batch_optimization(processor, documents)
    
    # Error handling demo
    error_results = demo_error_handling_and_recovery(processor, documents)
    
    # Resource management demo
    demo_resource_management(processor, documents)
    
    # Export results
    print("\n4️⃣ Exporting Results...")
    export_file = export_batch_results(sequential_results, parallel_results, optimization_results, error_results)
    
    # Final recommendations
    print("\n🎉 Batch Processing Demo Complete!")
    print("=" * 40)
    print("💡 Key insights for batch processing:")
    print("   🚀 Use parallel processing for large document sets")
    print("   ⚡ Optimize chunking parameters for your use case")
    print("   🛡️  Implement robust error handling and recovery")
    print("   💾 Monitor and manage resource usage")
    print("   📊 Track performance metrics for optimization")
    
    if export_file:
        print(f"\n📁 Detailed results exported to: {export_file}")
    
    print("\n📚 Explore other demos:")
    print("   - Basic workflow: python demos/01_basic_workflow_demo.py")
    print("   - CLI usage: python demos/02_cli_demo.py")
    print("   - Python API: python demos/03_api_demo.py")
    print("   - Performance: python demos/09_performance_benchmarking_demo.py")
    print("   - Usage analytics: python demos/14_usage_analytics_demo.py")

if __name__ == "__main__":
    main()
