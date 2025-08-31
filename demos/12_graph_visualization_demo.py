#!/usr/bin/env python3
"""
EchoGem Graph Visualization Demo

This demo showcases the interactive graph visualization features:
- Loading and displaying graph data
- Interactive exploration
- Layout switching
- Data export
- Custom visualization options
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from graphe import GraphVisualizer
from processor import Processor
from models import ChunkingOptions, QueryOptions

def create_sample_data_for_graph():
    """Create sample data to generate a rich graph for visualization"""
    print("📝 Creating sample data for graph visualization...")
    
    # Create a comprehensive transcript about AI and machine learning
    transcript = """
    Artificial Intelligence and Machine Learning: A Comprehensive Overview
    
    Introduction to AI
    Artificial Intelligence (AI) represents the pinnacle of computer science, enabling machines to perform tasks that traditionally required human intelligence. The field encompasses machine learning, natural language processing, computer vision, and robotics. AI systems can now understand, learn, reason, and interact with humans in increasingly sophisticated ways.
    
    Machine Learning Fundamentals
    Machine learning is a subset of AI that focuses on developing algorithms that can learn from and make predictions on data. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to teach models to make predictions, while unsupervised learning finds hidden patterns in unlabeled data. Reinforcement learning involves training agents to make decisions through trial and error.
    
    Deep Learning Revolution
    Deep learning, a subset of machine learning, has revolutionized AI by using artificial neural networks with multiple layers. These networks can automatically learn hierarchical representations of data, leading to breakthroughs in image recognition, speech processing, and natural language understanding. Convolutional neural networks (CNNs) excel at processing visual data, while recurrent neural networks (RNNs) are particularly effective for sequential data like text and speech.
    
    Natural Language Processing
    Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. Modern NLP systems use transformer architectures like BERT and GPT to achieve remarkable performance on tasks like machine translation, sentiment analysis, and question answering. These models can understand context, handle ambiguity, and generate coherent text that closely resembles human writing.
    
    Computer Vision Applications
    Computer vision allows machines to interpret and understand visual information from the world. Applications include facial recognition, autonomous vehicles, medical image analysis, and industrial quality control. Deep learning models have achieved superhuman performance on many computer vision tasks, enabling new applications that were previously impossible.
    
    AI Ethics and Responsibility
    As AI systems become more powerful and pervasive, ethical considerations become increasingly important. Key concerns include bias and fairness, transparency and explainability, privacy and security, and the potential for job displacement. Responsible AI development requires careful consideration of these issues and the implementation of appropriate safeguards and governance frameworks.
    
    Future of AI
    The future of AI holds tremendous promise for solving complex problems and improving human lives. Areas of active research include artificial general intelligence (AGI), quantum machine learning, and AI-human collaboration. However, realizing this potential requires addressing current limitations and ensuring that AI development aligns with human values and societal goals.
    
    AI in Healthcare
    AI is transforming healthcare through applications in medical diagnosis, drug discovery, personalized medicine, and patient monitoring. Machine learning algorithms can analyze medical images to detect diseases earlier and more accurately than human doctors. AI systems can also predict patient outcomes, optimize treatment plans, and accelerate drug development processes.
    
    AI in Education
    AI-powered educational systems can provide personalized learning experiences tailored to individual student needs and learning styles. These systems can identify areas where students struggle, provide targeted support, and track progress over time. AI tutors can offer 24/7 assistance, making quality education more accessible to people around the world.
    
    AI in Transportation
    Autonomous vehicles represent one of the most visible applications of AI in transportation. These systems use sophisticated sensors, machine learning algorithms, and real-time decision-making to navigate roads safely and efficiently. AI is also being used to optimize traffic flow, reduce congestion, and improve public transportation systems.
    
    Challenges and Limitations
    Despite remarkable progress, AI still faces significant challenges. Current systems lack common sense reasoning, struggle with transfer learning across domains, and require large amounts of training data. AI systems can also be brittle and fail in unexpected ways, raising concerns about safety and reliability in critical applications.
    
    AI Governance and Policy
    Effective governance of AI development and deployment is essential for maximizing benefits while minimizing risks. This includes establishing appropriate regulations, standards, and oversight mechanisms. International cooperation is particularly important given the global nature of AI technology and its potential impact on society.
    
    Human-AI Collaboration
    The most promising approach to AI development involves human-AI collaboration rather than replacement. AI systems can augment human capabilities, handling routine tasks while humans focus on creative problem-solving and decision-making. This collaborative approach combines the strengths of both human and artificial intelligence.
    
    Conclusion
    Artificial Intelligence represents one of the most transformative technologies of our time, with the potential to revolutionize virtually every aspect of human society. While significant challenges remain, the continued development of responsible AI systems offers enormous opportunities for improving human lives and addressing some of the world's most pressing problems.
    """
    
    return transcript

def demo_graph_creation():
    """Demonstrate creating a graph from sample data"""
    print("🎨 Graph Creation Demo")
    print("=" * 30)
    
    # Initialize processor
    print("1️⃣ Initializing EchoGem Processor...")
    try:
        processor = Processor()
        print("   ✅ Processor initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize processor: {e}")
        print("   💡 Make sure your API keys are set correctly")
        return None
    
    # Process sample transcript
    print("\n2️⃣ Processing sample transcript...")
    transcript = create_sample_data_for_graph()
    
    chunking_options = ChunkingOptions(
        max_chunk_size=400,
        overlap=75,
        semantic_chunking=True
    )
    
    try:
        start_time = time.time()
        response = processor.process_transcript(transcript, chunking_options=chunking_options)
        processing_time = time.time() - start_time
        
        print(f"   ✅ Created {len(response.chunks)} chunks in {processing_time:.2f}s")
        print(f"   📊 Average chunk size: {sum(len(c.content) for c in response.chunks) // len(response.chunks)} characters")
        
        # Ask some questions to generate Q&A pairs
        print("\n3️⃣ Generating Q&A pairs for graph...")
        questions = [
            "What is artificial intelligence?",
            "What are the main types of machine learning?",
            "How does deep learning work?",
            "What are the applications of AI in healthcare?",
            "What are the ethical concerns with AI?",
            "What is the future of AI?",
            "How does AI impact education?",
            "What are the challenges in AI development?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"   🤔 Question {i}: {question[:50]}...")
            try:
                result = processor.query(question, QueryOptions(max_chunks=3))
                print(f"      ✅ Answered in {result.query_time:.2f}s")
            except Exception as e:
                print(f"      ❌ Failed: {e}")
        
        return processor
        
    except Exception as e:
        print(f"   ❌ Processing failed: {e}")
        return None

def demo_graph_visualization():
    """Demonstrate graph visualization features"""
    print("\n🎨 Graph Visualization Demo")
    print("=" * 35)
    
    try:
        # Create visualizer
        print("   🎨 Creating GraphVisualizer...")
        visualizer = GraphVisualizer()
        print("   ✅ GraphVisualizer created successfully")
        
        # Check if we have data to visualize
        if not visualizer.nodes:
            print("   ⚠️  No data available for visualization")
            print("   💡 Process some transcripts first to see the graph")
            return visualizer
        
        print(f"   📊 Graph contains {len(visualizer.nodes)} nodes")
        
        # Analyze graph structure
        chunk_nodes = [n for n in visualizer.nodes.values() if n.node_type == 'chunk']
        qa_nodes = [n for n in visualizer.nodes.values() if n.node_type == 'qa_pair']
        
        print(f"      📦 Chunk nodes: {len(chunk_nodes)}")
        print(f"      💬 Q&A nodes: {len(qa_nodes)}")
        
        # Show some sample nodes
        if chunk_nodes:
            print(f"\n   🔍 Sample chunk nodes:")
            for i, node in enumerate(chunk_nodes[:3]):
                print(f"      {i+1}. {node.title[:60]}...")
                print(f"         Keywords: {', '.join(node.keywords[:3])}")
                print(f"         Usage count: {node.usage_count}")
        
        if qa_nodes:
            print(f"\n   💬 Sample Q&A nodes:")
            for i, node in enumerate(qa_nodes[:3]):
                print(f"      {i+1}. Q: {node.title[:60]}...")
                print(f"         Usage count: {node.usage_count}")
        
        return visualizer
        
    except ImportError:
        print("   ❌ GraphVisualizer not available")
        print("   💡 Install pygame: pip install pygame")
        return None
    except Exception as e:
        print(f"   ❌ Visualization failed: {e}")
        return None

def demo_graph_export(visualizer):
    """Demonstrate graph export functionality"""
    print("\n📤 Graph Export Demo")
    print("=" * 25)
    
    if not visualizer:
        print("   ⚠️  No visualizer available")
        return
    
    # Export graph data
    export_filename = f"demo_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        print(f"   📁 Exporting graph to: {export_filename}")
        visualizer.export_graph(export_filename)
        print("   ✅ Graph exported successfully")
        
        # Show export summary
        with open(export_filename, 'r') as f:
            export_data = json.load(f)
        
        print(f"   📊 Export summary:")
        print(f"      Nodes: {len(export_data.get('nodes', []))}")
        print(f"      Edges: {len(export_data.get('edges', []))}")
        print(f"      Metadata keys: {list(export_data.keys())}")
        
        # Show sample node structure
        if export_data.get('nodes'):
            sample_node = export_data['nodes'][0]
            print(f"\n   🔍 Sample node structure:")
            for key, value in sample_node.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"      {key}: {value[:100]}...")
                else:
                    print(f"      {key}: {value}")
        
        return export_filename
        
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        return None

def demo_graph_analysis(visualizer):
    """Demonstrate graph analysis features"""
    print("\n🔍 Graph Analysis Demo")
    print("=" * 25)
    
    if not visualizer:
        print("   ⚠️  No visualizer available")
        return
    
    print(f"   📊 Graph Statistics:")
    print(f"      Total nodes: {len(visualizer.nodes)}")
    
    # Node type distribution
    node_types = {}
    for node in visualizer.nodes.values():
        node_type = node.node_type
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    for node_type, count in node_types.items():
        print(f"      {node_type}: {count}")
    
    # Edge analysis
    if hasattr(visualizer, 'edges'):
        print(f"      Total edges: {len(visualizer.edges)}")
        
        # Edge type distribution
        edge_types = {}
        for edge in visualizer.edges:
            edge_type = edge.edge_type
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        for edge_type, count in edge_types.items():
            print(f"      {edge_type} edges: {count}")
    
    # Usage analysis
    print(f"\n   📈 Usage Analysis:")
    usage_counts = [node.usage_count for node in visualizer.nodes.values()]
    if usage_counts:
        avg_usage = sum(usage_counts) / len(usage_counts)
        max_usage = max(usage_counts)
        min_usage = min(usage_counts)
        
        print(f"      Average usage: {avg_usage:.1f}")
        print(f"      Maximum usage: {max_usage}")
        print(f"      Minimum usage: {min_usage}")
        
        # Most used nodes
        most_used = sorted(visualizer.nodes.values(), key=lambda x: x.usage_count, reverse=True)[:3]
        print(f"\n      🔥 Most used nodes:")
        for i, node in enumerate(most_used, 1):
            print(f"         {i}. {node.title[:50]}... (used {node.usage_count} times)")
    
    # Keyword analysis
    print(f"\n   🏷️  Keyword Analysis:")
    all_keywords = []
    for node in visualizer.nodes.values():
        all_keywords.extend(node.keywords)
    
    if all_keywords:
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Top keywords
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"      Top keywords:")
        for keyword, count in top_keywords:
            print(f"         {keyword}: {count} occurrences")

def demo_interactive_features():
    """Demonstrate interactive graph features"""
    print("\n🎮 Interactive Features Demo")
    print("=" * 35)
    
    print("   🎨 Interactive Graph Visualization")
    print("   💡 This will open a Pygame window with your graph")
    print()
    
    print("   🎮 Controls:")
    print("      - Mouse: Drag nodes, scroll to zoom")
    print("      - L: Toggle layout (force-directed, circular, hierarchical)")
    print("      - S: Save screenshot")
    print("      - E: Export graph data")
    print("      - ESC: Exit")
    print()
    
    print("   🔧 Layout Options:")
    print("      - Force-directed: Natural physics-based layout")
    print("      - Circular: Organized in concentric circles")
    print("      - Hierarchical: Tree-like structure")
    print()
    
    print("   📊 Visualization Features:")
    print("      - Node colors indicate type and usage")
    print("      - Edge thickness shows relationship strength")
    print("      - Hover over nodes for details")
    print("      - Click nodes to select and inspect")
    print()
    
    print("   🚀 To launch interactive visualization:")
    print("      echogem graph")
    print("   or")
    print("      python -c \"from graphe import GraphVisualizer; GraphVisualizer().run()\"")

def demo_customization_options():
    """Demonstrate graph customization options"""
    print("\n⚙️  Customization Options Demo")
    print("=" * 35)
    
    print("   🎨 Customization Options:")
    print()
    
    print("   📏 Screen Size:")
    print("      echogem graph --width 1600 --height 1000")
    print("      echogem graph --width 1920 --height 1080")
    print()
    
    print("   📁 Custom Data Source:")
    print("      echogem graph --usage-cache my_custom_cache.csv")
    print()
    
    print("   📤 Export Options:")
    print("      echogem graph --export my_analysis.json")
    print()
    
    print("   🔧 Programmatic Customization:")
    print("      from graphe import GraphVisualizer")
    print("      visualizer = GraphVisualizer(")
    print("          screen_width=1400,")
    print("          screen_height=900,")
    print("          usage_cache_path='custom_cache.csv'")
    print("      )")
    print("      visualizer.run()")
    print()
    
    print("   🎯 Advanced Features:")
    print("      - Custom node colors and sizes")
    print("      - Filtered views (by type, usage, etc.)")
    print("      - Animated transitions between layouts")
    print("      - Search and highlight functionality")

def main():
    """Main graph visualization demo function"""
    print("🎯 EchoGem Graph Visualization Demonstration")
    print("=" * 70)
    print("This demo showcases the interactive graph visualization features!")
    print()
    
    # Run graph creation demo
    processor = demo_graph_creation()
    if not processor:
        print("\n❌ Graph creation failed. Please check your setup and try again.")
        return
    
    # Run visualization demo
    visualizer = demo_graph_visualization()
    
    # Run export demo
    export_file = demo_graph_export(visualizer)
    
    # Run analysis demo
    demo_graph_analysis(visualizer)
    
    # Show interactive features
    demo_interactive_features()
    
    # Show customization options
    demo_customization_options()
    
    # Final recommendations
    print("\n🎉 Graph Visualization Demo Complete!")
    print("=" * 40)
    print("💡 Key features demonstrated:")
    print("   🎨 Interactive graph visualization")
    print("   📊 Data analysis and insights")
    print("   📤 Export and sharing capabilities")
    print("   🎮 Interactive exploration")
    print("   ⚙️  Customization options")
    
    if export_file:
        print(f"\n📁 Graph data exported to: {export_file}")
    
    print("\n🚀 Next steps:")
    print("   1. Launch interactive visualization: echogem graph")
    print("   2. Explore your data visually")
    print("   3. Export insights for analysis")
    print("   4. Customize the visualization for your needs")
    
    print("\n📚 Explore other demos:")
    print("   - Basic workflow: python demos/01_basic_workflow_demo.py")
    print("   - CLI usage: python demos/02_cli_demo.py")
    print("   - Python API: python demos/03_api_demo.py")
    print("   - Performance: python demos/09_performance_benchmarking_demo.py")

if __name__ == "__main__":
    main()
