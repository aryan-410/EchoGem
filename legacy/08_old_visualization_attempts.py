# Legacy: Old Visualization Attempts (v0.1.0-rc6)
# Different visualization approaches tested before pygame
# Replaced by current pygame-based GraphVisualizer in v0.2.0

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
import numpy as np

class MatplotlibVisualizer:
    """Matplotlib-based graph visualization - static but simple"""
    
    def __init__(self, nodes: List[Dict], edges: List[Dict]):
        self.nodes = nodes
        self.edges = edges
        self.graph = nx.Graph()
        self._build_graph()
    
    def _build_graph(self):
        """Build NetworkX graph from data"""
        # Add nodes
        for node in self.nodes:
            self.graph.add_node(
                node['id'], 
                pos=(node.get('x', 0), node.get('y', 0)),
                title=node.get('title', ''),
                node_type=node.get('node_type', 'unknown')
            )
        
        # Add edges
        for edge in self.edges:
            self.graph.add_edge(
                edge['source'], 
                edge['target'], 
                weight=edge.get('weight', 1.0)
            )
    
    def visualize(self, layout='spring', save_path=None):
        """Create static visualization"""
        plt.figure(figsize=(12, 8))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data['node_type'] == 'chunk':
                node_colors.append('lightblue')
                node_sizes.append(300)
            else:
                node_colors.append('lightgreen')
                node_sizes.append(400)
        
        nx.draw_networkx_nodes(
            self.graph, pos, 
            node_color=node_colors, 
            node_size=node_sizes
        )
        
        # Draw edges
        edge_weights = [self.graph[u][v]['weight'] for u, v in self.graph.edges()]
        nx.draw_networkx_edges(
            self.graph, pos, 
            width=[w * 2 for w in edge_weights],
            alpha=0.6
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos, 
            font_size=8,
            font_weight='bold'
        )
        
        plt.title("EchoGem Information Graph")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

# TESTING RESULTS:
# - Matplotlib: Good for static plots, no interactivity
# - Plotly: Interactive but web-based, complex setup
# - Terminal: Simple but very limited
# - HTML: Portable but requires browser
# - Tkinter: Built-in but limited graphics capabilities

# REPLACED BY: Pygame-based GraphVisualizer for best balance of features and simplicity
