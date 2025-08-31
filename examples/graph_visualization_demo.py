#!/usr/bin/env python3
"""
EchoGem Graph Visualization Demo
================================

This script demonstrates how to use EchoGem's graph visualization feature
to explore your transcript chunks and their relationships.

Features demonstrated:
- Interactive graph visualization
- Multiple layout algorithms
- Export functionality
- Usage statistics
- Node and edge exploration
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path so we can import echogem modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphe import GraphVisualizer


def demo_basic_visualization():
    """Demonstrate basic graph visualization"""
    print("🎯 Basic Graph Visualization Demo")
    print("=" * 40)
    
    usage_cache_path = "usage_cache_store.csv"
    
    if not os.path.exists(usage_cache_path):
        print(f"❌ Usage cache not found: {usage_cache_path}")
        print("💡 To create usage data, first process a transcript:")
        print("   echogem process your_transcript.txt")
        print("   echogem query 'What is the main topic?'")
        return None
    
    print(f"✅ Found usage cache: {usage_cache_path}")
    
    # Create visualizer with default settings
    visualizer = GraphVisualizer(
        usage_cache_path=usage_cache_path,
        screen_width=1200,
        screen_height=800
    )
    
    print(f"📊 Loaded {len(visualizer.nodes)} nodes and {len(visualizer.edges)} edges")
    
    if len(visualizer.nodes) == 0:
        print("⚠️  No nodes found. Make sure you have processed transcripts and made queries.")
        return None
    
    return visualizer


def demo_export_functionality(visualizer):
    """Demonstrate graph export functionality"""
    print("\n💾 Graph Export Demo")
    print("=" * 25)
    
    # Export to different formats
    export_files = [
        "demo_graph_export.json",
        "demo_graph_minimal.json",
        "demo_graph_detailed.json"
    ]
    
    for export_file in export_files:
        print(f"📁 Exporting to {export_file}...")
        visualizer.export_graph(export_file)
        
        # Show file size
        if os.path.exists(export_file):
            size = os.path.getsize(export_file)
            print(f"   ✅ Exported successfully ({size} bytes)")
        else:
            print(f"   ❌ Export failed")
    
    # Show sample of exported data
    if os.path.exists("demo_graph_export.json"):
        print("\n📋 Sample exported data structure:")
        with open("demo_graph_export.json", 'r') as f:
            data = json.load(f)
            print(f"   Nodes: {len(data.get('nodes', []))}")
            print(f"   Edges: {len(data.get('edges', []))}")
            print(f"   Metadata: {list(data.keys())}")


def demo_interactive_features(visualizer):
    """Demonstrate interactive features"""
    print("\n🎮 Interactive Features Demo")
    print("=" * 30)
    
    print("Available controls:")
    print("  🖱️  Mouse Controls:")
    print("     - Left click: Select nodes")
    print("     - Drag: Move nodes around")
    print("     - Hover: See node details")
    
    print("\n  ⌨️  Keyboard Controls:")
    print("     - SPACE: Cycle through layouts")
    print("     - L: Toggle node labels")
    print("     - E: Toggle edge display")
    print("     - U: Toggle usage statistics")
    print("     - ESC: Exit visualization")
    
    print("\n  🔄 Layout Modes:")
    print("     - Force-directed: Natural physics-based layout")
    print("     - Circular: Organized circular arrangement")
    print("     - Hierarchical: Tree-like structure")
    
    print("\n  📊 Display Options:")
    print("     - Node colors: Blue (chunks), Green (Q&A pairs)")
    print("     - Edge colors: Gray (similarity), Orange (usage)")
    print("     - Node size: Based on usage count")


def demo_usage_statistics(visualizer):
    """Demonstrate usage statistics"""
    print("\n📈 Usage Statistics Demo")
    print("=" * 25)
    
    if not visualizer.nodes:
        print("No nodes to analyze")
        return
    
    # Analyze node types
    chunk_nodes = [n for n in visualizer.nodes.values() if n.node_type == 'chunk']
    qa_nodes = [n for n in visualizer.nodes.values() if n.node_type == 'qa_pair']
    
    print(f"📊 Node Distribution:")
    print(f"   Chunks: {len(chunk_nodes)}")
    print(f"   Q&A Pairs: {len(qa_nodes)}")
    
    # Show most used chunks
    if chunk_nodes:
        most_used_chunks = sorted(chunk_nodes, key=lambda x: x.usage_count, reverse=True)[:3]
        print(f"\n🔥 Most Used Chunks:")
        for i, chunk in enumerate(most_used_chunks, 1):
            print(f"   {i}. '{chunk.title[:50]}...' (used {chunk.usage_count} times)")
    
    # Show recent activity
    recent_nodes = [n for n in visualizer.nodes.values() if n.last_used]
    if recent_nodes:
        recent_nodes.sort(key=lambda x: x.last_used or datetime.min, reverse=True)
        print(f"\n⏰ Recent Activity:")
        for i, node in enumerate(recent_nodes[:3], 1):
            time_ago = "recently" if node.last_used else "never"
            print(f"   {i}. {node.node_type}: '{node.title[:40]}...' ({time_ago})")


def demo_customization(visualizer):
    """Demonstrate customization options"""
    print("\n🎨 Customization Demo")
    print("=" * 22)
    
    print("You can customize the visualization by:")
    print("  1. Adjusting window size:")
    print("     echogem graph --width 1600 --height 1000")
    
    print("  2. Using different usage cache files:")
    print("     echogem graph --usage-cache custom_cache.csv")
    
    print("  3. Exporting to specific files:")
    print("     echogem graph --export my_analysis.json")
    
    print("  4. Combining options:")
    print("     echogem graph --width 1920 --height 1080 --export large_graph.json")


def main():
    """Main demo function"""
    print("🚀 EchoGem Graph Visualization Demo")
    print("=" * 50)
    print("This demo shows you how to explore your transcript data visually!")
    print()
    
    # Check if we have data to work with
    visualizer = demo_basic_visualization()
    if not visualizer:
        print("\n💡 To get started:")
        print("   1. Install EchoGem: pip install -e .")
        print("   2. Process a transcript: echogem process transcript.txt")
        print("   3. Ask some questions: echogem query 'What is this about?'")
        print("   4. Run this demo again!")
        return
    
    # Run all demos
    demo_export_functionality(visualizer)
    demo_interactive_features(visualizer)
    demo_usage_statistics(visualizer)
    demo_customization(visualizer)
    
    print("\n🎯 Ready to Launch Interactive Visualization!")
    print("=" * 45)
    print("To start the interactive graph visualization, run:")
    print("   echogem graph")
    print()
    print("Or customize the experience:")
    print("   echogem graph --width 1600 --height 1000")
    print("   echogem graph --export my_analysis.json")
    print()
    print("💡 Pro tip: The interactive visualization is great for:")
    print("   - Exploring relationships between chunks")
    print("   - Understanding which information is most relevant")
    print("   - Visualizing your knowledge base structure")
    print("   - Sharing insights with others")


if __name__ == "__main__":
    main()
