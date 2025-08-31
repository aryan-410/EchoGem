#!/usr/bin/env python3
"""
Example: Graph Visualization with EchoGem

This example demonstrates how to use the interactive graph visualization
to explore relationships between chunks and prompt-answer pairs.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphe import GraphVisualizer


def main():
    """Demonstrate graph visualization features"""
    print("🎯 EchoGem Graph Visualization Example")
    print("=" * 50)
    
    # Check if usage cache exists
    usage_cache_path = "usage_cache_store.csv"
    if not os.path.exists(usage_cache_path):
        print(f"❌ Usage cache not found: {usage_cache_path}")
        print("Please process a transcript first using:")
        print("  py -m echogem.cli process your_transcript.txt")
        return
    
    print(f"✅ Found usage cache: {usage_cache_path}")
    
    # Create visualizer
    print("\n🚀 Creating graph visualizer...")
    visualizer = GraphVisualizer(
        usage_cache_path=usage_cache_path,
        screen_width=1200,
        screen_height=800
    )
    
    print(f"📊 Loaded {len(visualizer.nodes)} nodes and {len(visualizer.edges)} edges")
    
    # Show available commands
    print("\n🎮 Available Controls:")
    print("  Mouse: Drag nodes, click to select")
    print("  Space: Toggle layout modes (force/circular/hierarchical)")
    print("  L: Toggle node labels")
    print("  E: Toggle edge display")
    print("  U: Toggle usage statistics")
    print("  ESC: Exit visualization")
    
    # Export graph data first
    export_file = "example_graph_export.json"
    print(f"\n💾 Exporting graph data to {export_file}...")
    visualizer.export_graph(export_file)
    
    # Ask user if they want to run visualization
    print(f"\n🎯 Graph data exported to {export_file}")
    print("You can now:")
    print("1. Run interactive visualization: echogem graph")
    print("2. View exported data in external tools")
    print("3. Customize visualization parameters")
    
    # Show some statistics
    chunk_nodes = [n for n in visualizer.nodes.values() if n.node_type == "chunk"]
    qa_nodes = [n for n in visualizer.nodes.values() if n.node_type == "qa_pair"]
    
    print(f"\n📈 Graph Statistics:")
    print(f"  Chunks: {len(chunk_nodes)}")
    print(f"  Q&A Pairs: {len(qa_nodes)}")
    print(f"  Total Edges: {len(visualizer.edges)}")
    
    if chunk_nodes:
        avg_usage = sum(n.usage_count for n in chunk_nodes) / len(chunk_nodes)
        print(f"  Average chunk usage: {avg_usage:.1f}")
    
    # Show sample nodes
    if chunk_nodes:
        print(f"\n📝 Sample Chunks:")
        for i, chunk in enumerate(chunk_nodes[:3], 1):
            print(f"  {i}. {chunk.title}")
            print(f"     Keywords: {', '.join(chunk.keywords[:3])}")
            print(f"     Usage: {chunk.usage_count}")
    
    if qa_nodes:
        print(f"\n❓ Sample Q&A Pairs:")
        for i, qa in enumerate(qa_nodes[:3], 1):
            print(f"  {i}. {qa.title}")
            print(f"     Usage: {qa.usage_count}")
    
    print(f"\n🎯 To launch interactive visualization, run:")
    print(f"  echogem graph")
    print(f"  echogem graph --width 1600 --height 1000")
    print(f"  echogem graph --export custom_export.json")


if __name__ == "__main__":
    main()
