#!/usr/bin/env python3
"""
EchoGem Usage Analytics Demo

This demo showcases comprehensive usage pattern analysis:
- Usage statistics and trends
- Chunk popularity analysis
- Query pattern analysis
- Performance metrics
- Optimization insights
"""

import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from processor import Processor
from usage_cache import UsageCache
from models import ChunkingOptions, QueryOptions

def create_sample_usage_data(processor):
    """Create sample usage data by processing transcripts and asking questions"""
    print("📝 Creating sample usage data...")
    
    # Create sample transcripts
    transcripts = {
        "ai_overview": """
        Artificial Intelligence: A Comprehensive Overview
        
        Artificial Intelligence (AI) represents the pinnacle of computer science, enabling machines to perform tasks that traditionally required human intelligence. The field encompasses machine learning, natural language processing, computer vision, and robotics.
        
        Machine learning is a subset of AI that focuses on developing algorithms that can learn from and make predictions on data. There are three main types: supervised learning, unsupervised learning, and reinforcement learning.
        
        Deep learning uses artificial neural networks with multiple layers to automatically learn hierarchical representations of data. This has led to breakthroughs in image recognition, speech processing, and natural language understanding.
        
        Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. Modern NLP systems use transformer architectures to achieve remarkable performance on various language tasks.
        
        Computer vision allows machines to interpret visual information from the world. Applications include facial recognition, autonomous vehicles, medical image analysis, and industrial quality control.
        
        AI ethics and responsibility are crucial as systems become more powerful. Key concerns include bias and fairness, transparency and explainability, privacy and security, and potential job displacement.
        
        The future of AI holds tremendous promise for solving complex problems and improving human lives. Areas of active research include artificial general intelligence, quantum machine learning, and AI-human collaboration.
        """,
        
        "machine_learning_basics": """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from and make predictions on data. These algorithms build mathematical models based on sample data, known as training data, to make predictions or decisions without being explicitly programmed.
        
        There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to teach models to make predictions, while unsupervised learning finds hidden patterns in unlabeled data.
        
        Supervised learning algorithms include linear regression, logistic regression, decision trees, random forests, and support vector machines. These algorithms are used for classification and regression tasks.
        
        Unsupervised learning algorithms include clustering algorithms like K-means, hierarchical clustering, and DBSCAN. These algorithms are used for finding patterns and grouping similar data points.
        
        Reinforcement learning involves training agents to make decisions through trial and error. The agent learns by interacting with an environment and receiving rewards or penalties for its actions.
        
        Feature engineering is a crucial step in machine learning that involves selecting and transforming relevant features from raw data. Good features can significantly improve model performance.
        
        Model evaluation is essential for assessing the quality of machine learning models. Common metrics include accuracy, precision, recall, F1-score, and ROC-AUC for classification tasks.
        """,
        
        "data_science_practices": """
        Data Science Best Practices
        
        Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, machine learning, data analysis, and domain expertise.
        
        The data science workflow typically involves several stages: data collection, data cleaning and preprocessing, exploratory data analysis, feature engineering, model development, model evaluation, and deployment.
        
        Data collection involves gathering data from various sources such as databases, APIs, web scraping, sensors, and surveys. The quality and quantity of data significantly impact the success of data science projects.
        
        Data cleaning and preprocessing are critical steps that involve handling missing values, removing duplicates, dealing with outliers, and ensuring data consistency. Clean data leads to better models and more reliable insights.
        
        Exploratory data analysis (EDA) involves examining and visualizing data to understand patterns, relationships, and anomalies. EDA helps identify potential issues and guides the modeling approach.
        
        Feature engineering involves creating new features from existing data that can improve model performance. This includes feature selection, transformation, and creation of interaction terms.
        
        Model development involves selecting appropriate algorithms, tuning hyperparameters, and training models. The choice of algorithm depends on the problem type, data characteristics, and performance requirements.
        
        Model evaluation involves assessing model performance using appropriate metrics and validation techniques. Cross-validation, holdout sets, and multiple evaluation metrics are commonly used.
        """
    }
    
    # Process transcripts
    print("   📝 Processing transcripts...")
    chunking_options = ChunkingOptions(max_chunk_size=400, overlap=50, semantic_chunking=True)
    
    for transcript_name, content in transcripts.items():
        try:
            response = processor.process_transcript(content, chunking_options=chunking_options)
            print(f"      ✅ {transcript_name}: {len(response.chunks)} chunks created")
        except Exception as e:
            print(f"      ❌ {transcript_name}: {e}")
    
    # Ask questions to generate usage data
    print("   🤔 Generating usage data through queries...")
    questions = [
        "What is artificial intelligence?",
        "What are the main types of machine learning?",
        "How does deep learning work?",
        "What is natural language processing?",
        "What are the applications of computer vision?",
        "What are the ethical concerns with AI?",
        "What is supervised learning?",
        "What is unsupervised learning?",
        "What is reinforcement learning?",
        "What is feature engineering?",
        "What are the best practices in data science?",
        "How do you evaluate machine learning models?",
        "What is the data science workflow?",
        "What is exploratory data analysis?",
        "What are the challenges in data science?"
    ]
    
    for i, question in enumerate(questions, 1):
        try:
            result = processor.query(question, QueryOptions(max_chunks=3))
            print(f"      ✅ Question {i}: Answered in {result.query_time:.2f}s")
        except Exception as e:
            print(f"      ❌ Question {i}: {e}")
    
    print("   ✅ Sample usage data created successfully")

def demo_usage_statistics(usage_cache):
    """Demonstrate basic usage statistics"""
    print("\n📊 Basic Usage Statistics")
    print("=" * 30)
    
    try:
        stats = usage_cache.get_usage_statistics()
        
        print("   📈 Overall Statistics:")
        print(f"      Total chunks accessed: {stats.get('total_chunks_accessed', 0)}")
        print(f"      Total unique chunks: {stats.get('total_unique_chunks', 0)}")
        print(f"      Most used chunks: {len(stats.get('most_used_chunks', []))}")
        print(f"      Recent activity: {len(stats.get('recent_activity', []))}")
        
        # Show most used chunks
        most_used = stats.get('most_used_chunks', [])
        if most_used:
            print(f"\n   🔥 Top 5 Most Used Chunks:")
            for i, chunk_info in enumerate(most_used[:5], 1):
                chunk_id = chunk_info.get('chunk_id', 'Unknown')[:8]
                usage_count = chunk_info.get('usage_count', 0)
                last_accessed = chunk_info.get('last_accessed', 'Unknown')
                print(f"      {i}. {chunk_id} (used {usage_count} times, last: {last_accessed})")
        
        # Show recent activity
        recent = stats.get('recent_activity', [])
        if recent:
            print(f"\n   ⏰ Recent Activity (Last 10):")
            for i, activity in enumerate(recent[:10], 1):
                chunk_id = activity.get('chunk_id', 'Unknown')[:8]
                timestamp = activity.get('timestamp', 'Unknown')
                print(f"      {i}. {chunk_id} at {timestamp}")
        
        return stats
        
    except Exception as e:
        print(f"   ❌ Failed to get usage statistics: {e}")
        return None

def demo_usage_patterns(usage_cache):
    """Demonstrate usage pattern analysis"""
    print("\n🔍 Usage Pattern Analysis")
    print("=" * 30)
    
    try:
        stats = usage_cache.get_usage_statistics()
        
        # Analyze usage patterns
        print("   📊 Usage Pattern Analysis:")
        
        # Time-based patterns
        recent_activity = stats.get('recent_activity', [])
        if recent_activity:
            # Group by hour to see usage patterns
            hourly_usage = {}
            for activity in recent_activity:
                timestamp = activity.get('timestamp')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        hour = dt.hour
                        hourly_usage[hour] = hourly_usage.get(hour, 0) + 1
                    except:
                        continue
            
            if hourly_usage:
                print(f"      📅 Hourly Usage Patterns:")
                for hour in sorted(hourly_usage.keys()):
                    count = hourly_usage[hour]
                    bar = '█' * (count // 2)  # Simple bar chart
                    print(f"         {hour:02d}:00 - {count:2d} accesses {bar}")
        
        # Chunk type analysis
        most_used_chunks = stats.get('most_used_chunks', [])
        if most_used_chunks:
            # Analyze usage distribution
            usage_counts = [chunk.get('usage_count', 0) for chunk in most_used_chunks]
            if usage_counts:
                avg_usage = sum(usage_counts) / len(usage_counts)
                max_usage = max(usage_counts)
                min_usage = min(usage_counts)
                
                print(f"\n      📊 Usage Distribution:")
                print(f"         Average usage per chunk: {avg_usage:.1f}")
                print(f"         Maximum usage: {max_usage}")
                print(f"         Minimum usage: {min_usage}")
                
                # Usage categories
                high_usage = len([c for c in usage_counts if c > avg_usage * 1.5])
                medium_usage = len([c for c in usage_counts if avg_usage * 0.5 <= c <= avg_usage * 1.5])
                low_usage = len([c for c in usage_counts if c < avg_usage * 0.5])
                
                print(f"         High usage chunks: {high_usage}")
                print(f"         Medium usage chunks: {medium_usage}")
                print(f"         Low usage chunks: {low_usage}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Usage pattern analysis failed: {e}")
        return False

def demo_chunk_popularity_analysis(usage_cache):
    """Demonstrate chunk popularity analysis"""
    print("\n🏆 Chunk Popularity Analysis")
    print("=" * 35)
    
    try:
        stats = usage_cache.get_usage_statistics()
        most_used_chunks = stats.get('most_used_chunks', [])
        
        if not most_used_chunks:
            print("   ⚠️  No usage data available for analysis")
            return False
        
        print("   📊 Popularity Metrics:")
        
        # Calculate popularity scores
        usage_counts = [chunk.get('usage_count', 0) for chunk in most_used_chunks]
        max_usage = max(usage_counts)
        
        popularity_scores = []
        for chunk in most_used_chunks:
            usage_count = chunk.get('usage_count', 0)
            popularity_score = (usage_count / max_usage) * 100 if max_usage > 0 else 0
            popularity_scores.append((chunk, popularity_score))
        
        # Sort by popularity
        popularity_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"      🏆 Top 10 Most Popular Chunks:")
        for i, (chunk, score) in enumerate(popularity_scores[:10], 1):
            chunk_id = chunk.get('chunk_id', 'Unknown')[:8]
            usage_count = chunk.get('usage_count', 0)
            last_accessed = chunk.get('last_accessed', 'Unknown')
            print(f"         {i:2d}. {chunk_id} - {usage_count} uses ({score:.1f}% popularity)")
            print(f"             Last accessed: {last_accessed}")
        
        # Popularity distribution
        print(f"\n      📈 Popularity Distribution:")
        high_popularity = len([s for s in popularity_scores if s[1] >= 80])
        medium_popularity = len([s for s in popularity_scores if 20 <= s[1] < 80])
        low_popularity = len([s for s in popularity_scores if s[1] < 20])
        
        print(f"         High popularity (80%+): {high_popularity} chunks")
        print(f"         Medium popularity (20-80%): {medium_popularity} chunks")
        print(f"         Low popularity (<20%): {low_popularity} chunks")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Chunk popularity analysis failed: {e}")
        return False

def demo_query_pattern_analysis(processor):
    """Demonstrate query pattern analysis"""
    print("\n🔍 Query Pattern Analysis")
    print("=" * 30)
    
    # Analyze different types of queries
    query_categories = {
        "Definition": [
            "What is artificial intelligence?",
            "Define machine learning",
            "What is deep learning?",
            "Explain natural language processing"
        ],
        "Comparison": [
            "How do supervised and unsupervised learning differ?",
            "Compare machine learning and deep learning",
            "What are the differences between AI and ML?"
        ],
        "Application": [
            "What are the applications of AI?",
            "How is machine learning used in practice?",
            "What are real-world examples of deep learning?"
        ],
        "Technical": [
            "How does neural network training work?",
            "What is feature engineering?",
            "How do you evaluate model performance?"
        ],
        "Process": [
            "What is the data science workflow?",
            "How do you approach a machine learning project?",
            "What are the steps in model development?"
        ]
    }
    
    print("   📊 Query Category Analysis:")
    
    category_results = {}
    
    for category, queries in query_categories.items():
        print(f"\n      🏷️  {category} Queries:")
        category_times = []
        
        for query in queries:
            try:
                start_time = time.time()
                result = processor.query(query, QueryOptions(max_chunks=2))
                query_time = time.time() - start_time
                category_times.append(query_time)
                
                print(f"         '{query[:40]}...' - {query_time:.3f}s")
                
            except Exception as e:
                print(f"         '{query[:40]}...' - Failed: {e}")
        
        if category_times:
            avg_time = sum(category_times) / len(category_times)
            category_results[category] = {
                'avg_time': avg_time,
                'query_count': len(category_times)
            }
            print(f"         📊 Average time: {avg_time:.3f}s")
    
    # Overall analysis
    if category_results:
        print(f"\n      📈 Overall Query Performance:")
        fastest_category = min(category_results.items(), key=lambda x: x[1]['avg_time'])
        slowest_category = max(category_results.items(), key=lambda x: x[1]['avg_time'])
        
        print(f"         Fastest category: {fastest_category[0]} ({fastest_category[1]['avg_time']:.3f}s)")
        print(f"         Slowest category: {slowest_category[0]} ({slowest_category[1]['avg_time']:.3f}s)")
        
        # Performance recommendations
        print(f"\n      💡 Performance Insights:")
        if fastest_category[1]['avg_time'] < 1.0:
            print(f"         ✅ {fastest_category[0]} queries are performing well")
        if slowest_category[1]['avg_time'] > 3.0:
            print(f"         ⚠️  {slowest_category[0]} queries may need optimization")
    
    return category_results

def demo_performance_metrics(usage_cache, processor):
    """Demonstrate performance metrics analysis"""
    print("\n⚡ Performance Metrics Analysis")
    print("=" * 35)
    
    try:
        # Get usage statistics
        stats = usage_cache.get_usage_statistics()
        
        print("   📊 Performance Overview:")
        
        # Response time analysis
        recent_activity = stats.get('recent_activity', [])
        if recent_activity:
            print(f"      📈 Recent Activity Performance:")
            print(f"         Total recent accesses: {len(recent_activity)}")
            
            # Time-based performance
            if len(recent_activity) >= 2:
                first_access = recent_activity[-1].get('timestamp')
                last_access = recent_activity[0].get('timestamp')
                
                if first_access and last_access:
                    try:
                        first_dt = datetime.fromisoformat(first_access.replace('Z', '+00:00'))
                        last_dt = datetime.fromisoformat(last_access.replace('Z', '+00:00'))
                        time_span = last_dt - first_dt
                        accesses_per_hour = len(recent_activity) / (time_span.total_seconds() / 3600)
                        
                        print(f"         Time span: {time_span}")
                        print(f"         Access rate: {accesses_per_hour:.1f} per hour")
                        
                    except:
                        print(f"         Time span: Unable to calculate")
        
        # Chunk efficiency
        most_used_chunks = stats.get('most_used_chunks', [])
        if most_used_chunks:
            print(f"\n      💾 Chunk Efficiency:")
            total_usage = sum(chunk.get('usage_count', 0) for chunk in most_used_chunks)
            unique_chunks = len(most_used_chunks)
            
            if unique_chunks > 0:
                avg_usage_per_chunk = total_usage / unique_chunks
                print(f"         Total chunk accesses: {total_usage}")
                print(f"         Unique chunks accessed: {unique_chunks}")
                print(f"         Average usage per chunk: {avg_usage_per_chunk:.1f}")
                
                # Efficiency rating
                if avg_usage_per_chunk > 5:
                    print(f"         🟢 High efficiency - chunks are well-utilized")
                elif avg_usage_per_chunk > 2:
                    print(f"         🟡 Medium efficiency - moderate chunk utilization")
                else:
                    print(f"         🔴 Low efficiency - chunks may be underutilized")
        
        # Query performance test
        print(f"\n      🔍 Query Performance Test:")
        test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the applications of AI?"
        ]
        
        query_times = []
        for query in test_queries:
            try:
                start_time = time.time()
                result = processor.query(query, QueryOptions(max_chunks=2))
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                print(f"         '{query[:30]}...' - {query_time:.3f}s")
                
            except Exception as e:
                print(f"         '{query[:30]}...' - Failed: {e}")
        
        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
            print(f"\n         📊 Average query time: {avg_query_time:.3f}s")
            
            # Performance rating
            if avg_query_time < 1.0:
                print(f"         🟢 Excellent performance")
            elif avg_query_time < 2.0:
                print(f"         🟡 Good performance")
            elif avg_query_time < 3.0:
                print(f"         🟠 Acceptable performance")
            else:
                print(f"         🔴 Performance may need improvement")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance metrics analysis failed: {e}")
        return False

def demo_optimization_insights(usage_cache, processor):
    """Demonstrate optimization insights based on usage data"""
    print("\n💡 Optimization Insights")
    print("=" * 30)
    
    try:
        stats = usage_cache.get_usage_statistics()
        
        print("   🔍 Usage-Based Insights:")
        
        # Identify underutilized chunks
        most_used_chunks = stats.get('most_used_chunks', [])
        if most_used_chunks:
            usage_counts = [chunk.get('usage_count', 0) for chunk in most_used_chunks]
            avg_usage = sum(usage_counts) / len(usage_counts)
            
            underutilized = [chunk for chunk in most_used_chunks if chunk.get('usage_count', 0) < avg_usage * 0.5]
            overutilized = [chunk for chunk in most_used_chunks if chunk.get('usage_count', 0) > avg_usage * 2]
            
            print(f"      📊 Chunk Utilization Analysis:")
            print(f"         Average usage: {avg_usage:.1f}")
            print(f"         Underutilized chunks: {len(underutilized)}")
            print(f"         Overutilized chunks: {len(overutilized)}")
            
            if underutilized:
                print(f"\n         ⚠️  Underutilized chunks (potential for consolidation):")
                for chunk in underutilized[:3]:
                    chunk_id = chunk.get('chunk_id', 'Unknown')[:8]
                    usage_count = chunk.get('usage_count', 0)
                    print(f"            {chunk_id} (used {usage_count} times)")
            
            if overutilized:
                print(f"\n         🔥 Overutilized chunks (consider splitting):")
                for chunk in overutilized[:3]:
                    chunk_id = chunk.get('chunk_id', 'Unknown')[:8]
                    usage_count = chunk.get('usage_count', 0)
                    print(f"            {chunk_id} (used {usage_count} times)")
        
        # Query optimization insights
        print(f"\n      🔍 Query Optimization Insights:")
        
        # Test different query strategies
        test_query = "What is artificial intelligence and how does it work?"
        
        # Test with different chunk limits
        chunk_limits = [1, 2, 3, 5]
        for limit in chunk_limits:
            try:
                start_time = time.time()
                result = processor.query(test_query, QueryOptions(max_chunks=limit))
                query_time = time.time() - start_time
                
                print(f"         {limit} chunks: {query_time:.3f}s")
                
            except Exception as e:
                print(f"         {limit} chunks: Failed")
        
        # Recommendations
        print(f"\n      💡 Optimization Recommendations:")
        
        if most_used_chunks:
            usage_counts = [chunk.get('usage_count', 0) for chunk in most_used_chunks]
            if len(usage_counts) > 1:
                usage_variance = sum((x - sum(usage_counts)/len(usage_counts))**2 for x in usage_counts) / len(usage_counts)
                usage_std = usage_variance ** 0.5
                
                if usage_std > avg_usage * 0.5:
                    print(f"         📊 High usage variance detected - consider chunk size optimization")
                else:
                    print(f"         📊 Usage distribution is relatively even")
        
        print(f"         🔍 Monitor query response times for performance bottlenecks")
        print(f"         📦 Consider chunk size adjustments based on usage patterns")
        print(f"         🎯 Focus on high-value chunks that are frequently accessed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Optimization insights failed: {e}")
        return False

def export_usage_analytics(usage_cache, category_results):
    """Export usage analytics to JSON file"""
    print("\n📤 Exporting Usage Analytics")
    print("=" * 35)
    
    try:
        # Get comprehensive usage data
        stats = usage_cache.get_usage_statistics()
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_chunks_accessed': stats.get('total_chunks_accessed', 0),
                'total_unique_chunks': stats.get('total_unique_chunks', 0),
                'most_used_chunks_count': len(stats.get('most_used_chunks', [])),
                'recent_activity_count': len(stats.get('recent_activity', []))
            },
            'usage_statistics': stats,
            'query_performance': category_results,
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'analysis_type': 'comprehensive_usage_analytics'
            }
        }
        
        filename = f"usage_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"   ✅ Usage analytics exported to: {filename}")
        
        # Show export summary
        print(f"   📊 Export Summary:")
        print(f"      Total chunks: {export_data['summary']['total_chunks_accessed']}")
        print(f"      Unique chunks: {export_data['summary']['total_unique_chunks']}")
        print(f"      Query categories analyzed: {len(category_results)}")
        
        return filename
        
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        return None

def main():
    """Main usage analytics demo function"""
    print("🎯 EchoGem Usage Analytics Demonstration")
    print("=" * 70)
    print("This demo showcases comprehensive usage pattern analysis and insights!")
    print()
    
    # Initialize components
    print("1️⃣ Initializing EchoGem components...")
    try:
        processor = Processor()
        usage_cache = UsageCache()
        print("   ✅ Components initialized successfully")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        print("   💡 Make sure your API keys are set correctly")
        return
    
    # Create sample usage data
    print("\n2️⃣ Creating sample usage data...")
    create_sample_usage_data(processor)
    
    # Run analytics demos
    print("\n3️⃣ Running Usage Analytics...")
    stats = demo_usage_statistics(usage_cache)
    demo_usage_patterns(usage_cache)
    demo_chunk_popularity_analysis(usage_cache)
    category_results = demo_query_pattern_analysis(processor)
    demo_performance_metrics(usage_cache, processor)
    demo_optimization_insights(usage_cache, processor)
    
    # Export results
    print("\n4️⃣ Exporting Results...")
    export_file = export_usage_analytics(usage_cache, category_results)
    
    # Final recommendations
    print("\n🎉 Usage Analytics Demo Complete!")
    print("=" * 40)
    print("💡 Key insights for usage analytics:")
    print("   📊 Monitor chunk utilization patterns")
    print("   🔍 Analyze query performance by category")
    print("   ⚡ Track response times and bottlenecks")
    print("   💡 Use insights for optimization")
    print("   📈 Monitor usage trends over time")
    
    if export_file:
        print(f"\n📁 Detailed analytics exported to: {export_file}")
    
    print("\n📚 Explore other demos:")
    print("   - Basic workflow: python demos/01_basic_workflow_demo.py")
    print("   - CLI usage: python demos/02_cli_demo.py")
    print("   - Python API: python demos/03_api_demo.py")
    print("   - Performance: python demos/09_performance_benchmarking_demo.py")
    print("   - Graph visualization: python demos/12_graph_visualization_demo.py")

if __name__ == "__main__":
    main()
