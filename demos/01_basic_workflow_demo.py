#!/usr/bin/env python3
"""
EchoGem Basic Workflow Demo

This demo showcases the complete end-to-end workflow:
1. Processing a transcript
2. Chunking and storing
3. Asking questions
4. Analyzing responses
5. Visualizing the information flow
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from processor import Processor
from models import ChunkingOptions, QueryOptions
from usage_cache import UsageCache

def create_sample_transcript():
    """Create a sample transcript for demonstration"""
    transcript = """
    Welcome to the Tech Innovation Summit 2024. I'm Dr. Sarah Chen, and I'll be your host today.
    
    Our first speaker is Dr. Michael Rodriguez from Stanford University, who will discuss "The Future of Artificial Intelligence in Healthcare." Dr. Rodriguez has been leading research in medical AI for over 15 years and has published more than 200 papers on the subject.
    
    Dr. Rodriguez: Thank you, Dr. Chen. The intersection of AI and healthcare represents one of the most promising frontiers in modern medicine. We're seeing breakthroughs in early disease detection, personalized treatment plans, and drug discovery that were unimaginable just a decade ago.
    
    Let me share some specific examples. Our team at Stanford has developed an AI system that can detect early-stage lung cancer from CT scans with 94% accuracy, compared to 70% for human radiologists. This system has already been deployed in three major hospitals and has helped save hundreds of lives.
    
    Another breakthrough area is drug discovery. Traditional drug development takes 10-15 years and costs billions of dollars. Our AI models can now predict drug efficacy and safety in weeks, reducing development time by 60% and costs by 40%.
    
    However, we must address several challenges. Data privacy is paramount - we need robust systems to protect patient information while enabling AI learning. Regulatory frameworks must evolve to keep pace with technological advances. And we must ensure AI systems are explainable and trustworthy.
    
    Our next speaker is Dr. Emily Watson from Google Research, who will discuss "Large Language Models in Scientific Research." Dr. Watson has been instrumental in developing models that can read and understand scientific literature at scale.
    
    Dr. Watson: Thank you. Large language models are revolutionizing how we approach scientific research. These models can process millions of research papers, identify patterns, and generate hypotheses that humans might miss.
    
    For instance, our team discovered a potential new antibiotic by analyzing the relationship between chemical structures and antimicrobial properties across thousands of papers. The AI identified a compound that had been overlooked for 40 years.
    
    We're also seeing breakthroughs in protein folding prediction, climate modeling, and materials science. The key is combining the pattern recognition capabilities of LLMs with domain-specific knowledge and rigorous scientific validation.
    
    But we must be careful about over-reliance. These models are tools, not replacements for human expertise. They can help us ask better questions and explore new directions, but the interpretation and validation must come from human scientists.
    
    Our final speaker is Dr. James Thompson from Microsoft Research, discussing "The Future of Human-Computer Interaction." Dr. Thompson has been working on next-generation interfaces that blend physical and digital worlds.
    
    Dr. Thompson: The way we interact with computers is fundamentally changing. We're moving beyond screens and keyboards to interfaces that understand natural language, gesture, and even thought.
    
    Our research in brain-computer interfaces has shown that we can decode basic intentions from brain signals with 85% accuracy. While this technology is still in early stages, it could revolutionize how people with disabilities interact with technology.
    
    We're also exploring augmented reality interfaces that overlay digital information on the physical world. Imagine walking through a city and seeing real-time information about buildings, restaurants, and events projected onto your field of vision.
    
    The key challenge is making these interfaces intuitive and accessible. They must feel natural, like extensions of our own capabilities, rather than foreign tools we have to learn.
    
    Thank you all for your attention. We'll now open the floor for questions and discussion.
    """
    return transcript.strip()

def demo_basic_workflow():
    """Demonstrate the complete EchoGem workflow"""
    print("🚀 EchoGem Basic Workflow Demo")
    print("=" * 50)
    
    # Initialize processor
    print("\n1️⃣ Initializing EchoGem Processor...")
    try:
        processor = Processor()
        print("   ✅ Processor initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize processor: {e}")
        print("   💡 Make sure your API keys are set correctly")
        return None
    
    # Create sample transcript
    print("\n2️⃣ Creating sample transcript...")
    transcript = create_sample_transcript()
    print(f"   📝 Transcript created: {len(transcript)} characters")
    print(f"   📊 Estimated chunks: {len(transcript) // 500 + 1}")
    
    # Process transcript
    print("\n3️⃣ Processing transcript...")
    start_time = time.time()
    
    chunking_options = ChunkingOptions(
        max_chunk_size=500,
        overlap=50,
        semantic_chunking=True
    )
    
    try:
        chunk_response = processor.process_transcript(
            transcript, 
            chunking_options=chunking_options
        )
        processing_time = time.time() - start_time
        
        print(f"   ✅ Processing completed in {processing_time:.2f} seconds")
        print(f"   📦 Created {len(chunk_response.chunks)} chunks")
        print(f"   🎯 Average chunk size: {sum(len(c.content) for c in chunk_response.chunks) // len(chunk_response.chunks)} characters")
        
    except Exception as e:
        print(f"   ❌ Processing failed: {e}")
        return None
    
    # Ask questions
    print("\n4️⃣ Asking questions...")
    questions = [
        "What are the main topics discussed at the summit?",
        "What breakthroughs has Dr. Rodriguez achieved in medical AI?",
        "How are large language models changing scientific research?",
        "What are the challenges in brain-computer interfaces?",
        "What is the overall theme of the conference?"
    ]
    
    query_options = QueryOptions(
        show_chunks=True,
        show_prompt_answers=True,
        max_chunks=3
    )
    
    responses = []
    for i, question in enumerate(questions, 1):
        print(f"   🤔 Question {i}: {question}")
        
        start_time = time.time()
        try:
            result = processor.query(question, query_options=query_options)
            query_time = time.time() - start_time
            
            print(f"      ⏱️  Response time: {query_time:.2f}s")
            print(f"      📝 Answer: {result.answer[:100]}...")
            print(f"      🔗 Chunks used: {len(result.chunks_used)}")
            print(f"      💬 Prompt-answers: {len(result.prompt_answers_used)}")
            
            responses.append(result)
            
        except Exception as e:
            print(f"      ❌ Query failed: {e}")
    
    # Analyze usage patterns
    print("\n5️⃣ Analyzing usage patterns...")
    try:
        usage_cache = UsageCache()
        usage_stats = usage_cache.get_usage_statistics()
        
        print(f"   📊 Total chunks accessed: {usage_stats['total_chunks_accessed']}")
        print(f"   🔥 Most used chunks: {len(usage_stats['most_used_chunks'])}")
        print(f"   ⏰ Recent activity: {len(usage_stats['recent_activity'])}")
        
    except Exception as e:
        print(f"   ⚠️  Could not analyze usage: {e}")
    
    # Performance summary
    print("\n6️⃣ Performance Summary")
    print("=" * 30)
    print(f"   📈 Total processing time: {processing_time:.2f}s")
    print(f"   🎯 Average query time: {sum(r.query_time for r in responses if hasattr(r, 'query_time')) / len(responses):.2f}s")
    print(f"   📦 Chunks created: {len(chunk_response.chunks)}")
    print(f"   ❓ Questions answered: {len(responses)}")
    print(f"   💾 Memory efficiency: {len(transcript) / (1024 * 1024):.2f} MB input → {len(chunk_response.chunks)} searchable chunks")
    
    return processor, chunk_response, responses

def demo_visualization():
    """Demonstrate graph visualization"""
    print("\n7️⃣ Graph Visualization Demo")
    print("=" * 35)
    
    try:
        from graphe import GraphVisualizer
        
        print("   🎨 Launching interactive graph visualization...")
        print("   💡 This will open a Pygame window showing your information flow")
        print("   🎮 Controls:")
        print("      - Mouse: Drag nodes, scroll to zoom")
        print("      - L: Toggle layout (force-directed, circular, hierarchical)")
        print("      - S: Save screenshot")
        print("      - E: Export graph data")
        print("      - ESC: Exit")
        
        # Create visualizer
        visualizer = GraphVisualizer()
        
        # Export graph data for inspection
        export_file = "demo_workflow_graph.json"
        visualizer.export_graph(export_file)
        print(f"   📁 Graph data exported to: {export_file}")
        
        print("\n   🚀 To launch interactive visualization, run:")
        print("      echogem graph")
        print("   or")
        print("      python -c \"from graphe import GraphVisualizer; GraphVisualizer().run()\"")
        
    except ImportError:
        print("   ⚠️  Graph visualization not available")
        print("   💡 Install pygame: pip install pygame")
    except Exception as e:
        print(f"   ❌ Visualization failed: {e}")

def main():
    """Main demo function"""
    print("🎯 EchoGem Complete Workflow Demonstration")
    print("=" * 60)
    print("This demo will show you the full power of EchoGem!")
    print()
    
    # Run basic workflow
    result = demo_basic_workflow()
    if not result:
        print("\n❌ Demo failed. Please check your setup and try again.")
        return
    
    processor, chunk_response, responses = result
    
    # Show visualization options
    demo_visualization()
    
    # Final recommendations
    print("\n🎉 Demo Complete!")
    print("=" * 20)
    print("💡 Next steps:")
    print("   1. Try your own transcripts: echogem process your_file.txt")
    print("   2. Ask custom questions: echogem query 'Your question here'")
    print("   3. Explore the graph: echogem graph")
    print("   4. Check usage patterns: python -c \"from usage_cache import UsageCache; print(UsageCache().get_usage_statistics())\"")
    
    print("\n📚 Explore other demos:")
    print("   - Academic papers: python demos/04_academic_paper_demo.py")
    print("   - Performance testing: python demos/09_performance_benchmarking_demo.py")
    print("   - Custom chunking: python demos/10_custom_chunking_demo.py")

if __name__ == "__main__":
    main()
