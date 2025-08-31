"""
Graph visualization for EchoGem - shows information flow and relationships between chunks and Q&A pairs
"""

import os
import csv
import json
import hashlib
import math
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Install with: pip install pygame")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Install with: pip install numpy")

from .models import Chunk
from .usage_cache import UsageCache
from .prompt_answer_store import PromptAnswerVectorDB, PAPair


@dataclass
class GraphNode:
    """Node in the information graph"""
    node_id: str
    node_type: str  # 'chunk' or 'qa_pair'
    title: str
    content: str
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    timestamp_range: str = ""
    last_used: Optional[datetime] = None
    usage_count: int = 0
    x: float = 0.0
    y: float = 0.0
    radius: float = 15.0
    color: Tuple[int, int, int] = (100, 150, 255)
    selected: bool = False
    hover: bool = False


@dataclass
class GraphEdge:
    """Edge connecting nodes in the information graph"""
    source_id: str
    target_id: str
    weight: float = 1.0
    edge_type: str = "similarity"  # 'similarity', 'usage', 'temporal'
    color: Tuple[int, int, int] = (200, 200, 200)


class GraphVisualizer:
    """
    Interactive graph visualization for EchoGem information flow.
    
    Features:
    - Visual representation of chunks and Q&A pairs
    - Interactive node exploration
    - Relationship strength visualization
    - Usage pattern analysis
    - Temporal flow visualization
    """
    
    def __init__(
        self,
        usage_cache_path: str = "usage_cache_store.csv",
        pa_db: Optional[PromptAnswerVectorDB] = None,
        screen_width: int = 1200,
        screen_height: int = 800,
        node_spacing: float = 100.0,
        force_strength: float = 0.1
    ):
        """
        Initialize the graph visualizer
        
        Args:
            usage_cache_path: Path to usage cache CSV
            pa_db: Prompt-answer database instance
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            node_spacing: Minimum spacing between nodes
            force_strength: Strength of force-directed layout
        """
        self.usage_cache_path = usage_cache_path
        self.pa_db = pa_db
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.node_spacing = node_spacing
        self.force_strength = force_strength
        
        # Graph data
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.selected_node: Optional[GraphNode] = None
        self.dragging = False
        self.drag_offset = (0, 0)
        
        # Display settings
        self.show_labels = True
        self.show_edges = True
        self.show_usage_stats = True
        self.layout_mode = "force"  # 'force', 'circular', 'hierarchical'
        
        # Colors
        self.colors = {
            'chunk': (100, 150, 255),      # Blue for chunks
            'qa_pair': (255, 150, 100),    # Orange for Q&A pairs
            'selected': (255, 255, 0),     # Yellow for selected
            'hover': (255, 255, 255),      # White for hover
            'edge': (200, 200, 200),       # Gray for edges
            'background': (20, 20, 40),    # Dark blue background
            'text': (255, 255, 255),       # White text
            'ui': (100, 100, 150)          # UI elements
        }
        
        # Initialize pygame if available
        if PYGAME_AVAILABLE:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("EchoGem Information Graph")
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
        else:
            self.screen = None
            self.font = None
            self.small_font = None
        
        # Load data
        self._load_graph_data()
        
    def _load_graph_data(self) -> None:
        """Load chunks and Q&A pairs into the graph"""
        print("Loading graph data...")
        
        # Load chunks from usage cache
        try:
            usage_cache = UsageCache(self.usage_cache_path)
            chunks_data = usage_cache.get_all_chunks()
            
            for chunk_id, chunk_data in chunks_data.items():
                node = GraphNode(
                    node_id=chunk_id,
                    node_type="chunk",
                    title=chunk_data.get("title", f"Chunk {chunk_id[:8]}"),
                    content=chunk_data.get("content", ""),
                    keywords=self._parse_list_field(chunk_data.get("keywords", "[]")),
                    entities=self._parse_list_field(chunk_data.get("named_entities", "[]")),
                    timestamp_range=chunk_data.get("timestamp_range", ""),
                    last_used=self._parse_iso(chunk_data.get("last_used")),
                    usage_count=int(chunk_data.get("usage_count", 0)),
                    color=self.colors['chunk']
                )
                self.nodes[chunk_id] = node
                
        except Exception as e:
            print(f"Error loading chunks: {e}")
        
        # Load Q&A pairs if available
        if self.pa_db:
            try:
                # Get some sample Q&A pairs (this would need to be implemented in PA DB)
                print("Loading Q&A pairs...")
                # For now, we'll create some sample Q&A nodes
                self._create_sample_qa_nodes()
            except Exception as e:
                print(f"Error loading Q&A pairs: {e}")
        
        # Create edges based on relationships
        self._create_edges()
        
        # Initialize layout
        self._initialize_layout()
        
        print(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    def _create_sample_qa_nodes(self) -> None:
        """Create sample Q&A nodes for demonstration"""
        sample_qa = [
            ("What is the main topic?", "The main topic discussed is...", 5),
            ("Who are the speakers?", "The speakers include...", 3),
            ("What conclusions were reached?", "The key conclusions are...", 4),
        ]
        
        for i, (prompt, answer, usage) in enumerate(sample_qa):
            qa_id = f"qa_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
            node = GraphNode(
                node_id=qa_id,
                node_type="qa_pair",
                title=f"Q&A {i+1}",
                content=f"Q: {prompt}\nA: {answer}",
                keywords=prompt.lower().split()[:5],
                entities=[],
                timestamp_range="",
                last_used=datetime.now(timezone.utc),
                usage_count=usage,
                color=self.colors['qa_pair']
            )
            self.nodes[qa_id] = node
    
    def _create_edges(self) -> None:
        """Create edges between related nodes"""
        print("Creating edges...")
        
        # Create edges between chunks with similar keywords
        chunk_nodes = [n for n in self.nodes.values() if n.node_type == "chunk"]
        
        for i, node1 in enumerate(chunk_nodes):
            for j, node2 in enumerate(chunk_nodes[i+1:], i+1):
                # Calculate similarity based on keywords and entities
                similarity = self._calculate_similarity(node1, node2)
                
                if similarity > 0.3:  # Threshold for creating edges
                    edge = GraphEdge(
                        source_id=node1.node_id,
                        target_id=node2.node_id,
                        weight=similarity,
                        edge_type="similarity",
                        color=self._get_edge_color(similarity)
                    )
                    self.edges.append(edge)
        
        # Create edges between Q&A pairs and related chunks
        qa_nodes = [n for n in self.nodes.values() if n.node_type == "qa_pair"]
        
        for qa_node in qa_nodes:
            for chunk_node in chunk_nodes:
                # Find chunks that might answer the Q&A
                relevance = self._calculate_qa_chunk_relevance(qa_node, chunk_node)
                
                if relevance > 0.4:  # Threshold for Q&A-chunk connections
                    edge = GraphEdge(
                        source_id=qa_node.node_id,
                        target_id=chunk_node.node_id,
                        weight=relevance,
                        edge_type="relevance",
                        color=self._get_edge_color(relevance)
                    )
                    self.edges.append(edge)
    
    def _calculate_similarity(self, node1: GraphNode, node2: GraphNode) -> float:
        """Calculate similarity between two nodes"""
        # Keyword overlap
        keywords1 = set(node1.keywords)
        keywords2 = set(node2.keywords)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        keyword_sim = intersection / union if union > 0 else 0.0
        
        # Entity overlap
        entities1 = set(node1.entities)
        entities2 = set(node2.entities)
        
        if not entities1 or not entities2:
            entity_sim = 0.0
        else:
            intersection = len(entities1.intersection(entities2))
            union = len(entities1.union(entities2))
            entity_sim = intersection / union if union > 0 else 0.0
        
        # Usage pattern similarity
        usage_sim = 1.0 - abs(node1.usage_count - node2.usage_count) / max(1, max(node1.usage_count, node2.usage_count))
        
        # Weighted combination
        return 0.5 * keyword_sim + 0.3 * entity_sim + 0.2 * usage_sim
    
    def _calculate_qa_chunk_relevance(self, qa_node: GraphNode, chunk_node: GraphNode) -> float:
        """Calculate relevance between Q&A pair and chunk"""
        # Check if chunk keywords appear in the Q&A
        qa_text = qa_node.content.lower()
        chunk_keywords = [kw.lower() for kw in chunk_node.keywords]
        
        keyword_matches = sum(1 for kw in chunk_keywords if kw in qa_text)
        
        if not chunk_keywords:
            return 0.0
        
        return keyword_matches / len(chunk_keywords)
    
    def _get_edge_color(self, weight: float) -> Tuple[int, int, int]:
        """Get edge color based on weight"""
        intensity = int(255 * weight)
        return (intensity, intensity, intensity)
    
    def _initialize_layout(self) -> None:
        """Initialize node positions"""
        if self.layout_mode == "circular":
            self._circular_layout()
        elif self.layout_mode == "hierarchical":
            self._hierarchical_layout()
        else:
            self._force_directed_layout()
    
    def _circular_layout(self) -> None:
        """Arrange nodes in a circle"""
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        radius = min(self.screen_width, self.screen_height) / 3
        
        nodes = list(self.nodes.values())
        angle_step = 2 * math.pi / len(nodes) if nodes else 0
        
        for i, node in enumerate(nodes):
            angle = i * angle_step
            node.x = center_x + radius * math.cos(angle)
            node.y = center_y + radius * math.sin(angle)
    
    def _hierarchical_layout(self) -> None:
        """Arrange nodes in a hierarchical layout"""
        # Group by type
        chunks = [n for n in self.nodes.values() if n.node_type == "chunk"]
        qa_pairs = [n for n in self.nodes.values() if n.node_type == "qa_pair"]
        
        # Position chunks in top half
        chunk_cols = max(1, int(math.sqrt(len(chunks))))
        for i, chunk in enumerate(chunks):
            col = i % chunk_cols
            row = i // chunk_cols
            chunk.x = (col + 1) * (self.screen_width / (chunk_cols + 1))
            chunk.y = 100 + row * 80
        
        # Position Q&A pairs in bottom half
        qa_cols = max(1, int(math.sqrt(len(qa_pairs))))
        for i, qa in enumerate(qa_pairs):
            col = i % qa_cols
            row = i // qa_cols
            qa.x = (col + 1) * (self.screen_width / (qa_cols + 1))
            qa.y = self.screen_height - 200 + row * 80
    
    def _force_directed_layout(self) -> None:
        """Apply force-directed layout"""
        # Initialize random positions
        for node in self.nodes.values():
            node.x = random.uniform(50, self.screen_width - 50)
            node.y = random.uniform(50, self.screen_height - 50)
        
        # Apply forces
        for _ in range(100):  # Iterations
            # Repulsion between all nodes
            for node1 in self.nodes.values():
                fx, fy = 0, 0
                
                for node2 in self.nodes.values():
                    if node1.node_id == node2.node_id:
                        continue
                    
                    dx = node2.x - node1.x
                    dy = node2.y - node1.y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < self.node_spacing and distance > 0:
                        # Repulsion force
                        force = self.force_strength * (self.node_spacing - distance) / distance
                        fx -= force * dx
                        fy -= force * dy
                
                # Attraction from edges
                for edge in self.edges:
                    if edge.source_id == node1.node_id:
                        target = self.nodes.get(edge.target_id)
                        if target:
                            dx = target.x - node1.x
                            dy = target.y - node1.y
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance > 0:
                                force = self.force_strength * edge.weight * 0.1
                                fx += force * dx / distance
                                fy += force * dy / distance
                    
                    elif edge.target_id == node1.node_id:
                        source = self.nodes.get(edge.source_id)
                        if source:
                            dx = source.x - node1.x
                            dy = source.y - node1.y
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance > 0:
                                force = self.force_strength * edge.weight * 0.1
                                fx += force * dx / distance
                                fy += force * dy / distance
                
                # Apply forces
                node1.x += fx
                node1.y += fy
                
                # Keep nodes within bounds
                node1.x = max(50, min(self.screen_width - 50, node1.x))
                node1.y = max(50, min(self.screen_height - 50, node1.y))
    
    def _parse_list_field(self, value: Any) -> List[str]:
        """Parse list field from CSV data"""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return [v.strip() for v in value.split(',') if v.strip()]
        return []
    
    def _parse_iso(self, iso_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp string"""
        if not iso_str:
            return None
        try:
            return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        except:
            return None
    
    def run(self) -> None:
        """Run the interactive graph visualization"""
        if not PYGAME_AVAILABLE:
            print("Pygame not available. Cannot run interactive visualization.")
            return
        
        print("Starting graph visualization...")
        print("Controls:")
        print("  Mouse: Drag nodes, click to select")
        print("  Space: Toggle layout modes")
        print("  L: Toggle labels")
        print("  E: Toggle edges")
        print("  U: Toggle usage stats")
        print("  ESC: Exit")
        
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self._cycle_layout_mode()
                    elif event.key == pygame.K_l:
                        self.show_labels = not self.show_labels
                    elif event.key == pygame.K_e:
                        self.show_edges = not self.show_edges
                    elif event.key == pygame.K_u:
                        self.show_usage_stats = not self.show_usage_stats
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_down(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._handle_mouse_up(event)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event)
            
            # Update
            self._update()
            
            # Draw
            self._draw()
            
            # Cap frame rate
            clock.tick(60)
        
        pygame.quit()
    
    def _handle_mouse_down(self, event) -> None:
        """Handle mouse button down events"""
        if event.button == 1:  # Left click
            mouse_pos = pygame.mouse.get_pos()
            clicked_node = self._get_node_at_position(mouse_pos)
            
            if clicked_node:
                self.selected_node = clicked_node
                self.dragging = True
                self.drag_offset = (mouse_pos[0] - clicked_node.x, mouse_pos[1] - clicked_node.y)
            else:
                self.selected_node = None
    
    def _handle_mouse_up(self, event) -> None:
        """Handle mouse button up events"""
        if event.button == 1:  # Left click
            self.dragging = False
    
    def _handle_mouse_motion(self, event) -> None:
        """Handle mouse motion events"""
        if self.dragging and self.selected_node:
            mouse_pos = pygame.mouse.get_pos()
            self.selected_node.x = mouse_pos[0] - self.drag_offset[0]
            self.selected_node.y = mouse_pos[1] - self.drag_offset[1]
            
            # Keep within bounds
            self.selected_node.x = max(50, min(self.screen_width - 50, self.selected_node.x))
            self.selected_node.y = max(50, min(self.screen_height - 50, self.selected_node.y))
        
        # Update hover state
        mouse_pos = pygame.mouse.get_pos()
        for node in self.nodes.values():
            node.hover = self._is_point_in_node(mouse_pos, node)
    
    def _get_node_at_position(self, pos: Tuple[int, int]) -> Optional[GraphNode]:
        """Get node at given position"""
        for node in self.nodes.values():
            if self._is_point_in_node(pos, node):
                return node
        return None
    
    def _is_point_in_node(self, pos: Tuple[int, int], node: GraphNode) -> bool:
        """Check if point is inside node"""
        dx = pos[0] - node.x
        dy = pos[1] - node.y
        return dx*dx + dy*dy <= node.radius*node.radius
    
    def _cycle_layout_mode(self) -> None:
        """Cycle through layout modes"""
        modes = ["force", "circular", "hierarchical"]
        current_index = modes.index(self.layout_mode)
        self.layout_mode = modes[(current_index + 1) % len(modes)]
        print(f"Layout mode: {self.layout_mode}")
        self._initialize_layout()
    
    def _update(self) -> None:
        """Update graph state"""
        # Apply force-directed layout if enabled
        if self.layout_mode == "force":
            self._force_directed_layout()
    
    def _draw(self) -> None:
        """Draw the graph"""
        if not self.screen:
            return
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw edges
        if self.show_edges:
            self._draw_edges()
        
        # Draw nodes
        self._draw_nodes()
        
        # Draw UI
        self._draw_ui()
        
        # Update display
        pygame.display.flip()
    
    def _draw_edges(self) -> None:
        """Draw edges between nodes"""
        for edge in self.edges:
            source = self.nodes.get(edge.source_id)
            target = self.nodes.get(edge.target_id)
            
            if source and target:
                # Calculate edge color based on weight
                intensity = int(255 * edge.weight)
                color = (intensity, intensity, intensity)
                
                # Draw edge
                pygame.draw.line(
                    self.screen, color,
                    (int(source.x), int(source.y)),
                    (int(target.x), int(target.y)),
                    max(1, int(edge.weight * 3))
                )
    
    def _draw_nodes(self) -> None:
        """Draw all nodes"""
        for node in self.nodes.values():
            # Determine node color
            if node.selected:
                color = self.colors['selected']
            elif node.hover:
                color = self.colors['hover']
            else:
                color = node.color
            
            # Draw node
            pygame.draw.circle(
                self.screen, color,
                (int(node.x), int(node.y)),
                int(node.radius)
            )
            
            # Draw border
            pygame.draw.circle(
                self.screen, (255, 255, 255),
                (int(node.x), int(node.y)),
                int(node.radius), 2
            )
            
            # Draw labels
            if self.show_labels:
                self._draw_node_label(node)
    
    def _draw_node_label(self, node: GraphNode) -> None:
        """Draw label for a node"""
        if not self.font:
            return
        
        # Truncate title if too long
        title = node.title[:20] + "..." if len(node.title) > 20 else node.title
        
        # Render text
        text_surface = self.font.render(title, True, self.colors['text'])
        text_rect = text_surface.get_rect()
        
        # Position below node
        text_rect.centerx = int(node.x)
        text_rect.top = int(node.y + node.radius + 5)
        
        # Draw text
        self.screen.blit(text_surface, text_rect)
        
        # Draw usage count if enabled
        if self.show_usage_stats and node.usage_count > 0:
            usage_text = f"({node.usage_count})"
            usage_surface = self.small_font.render(usage_text, True, (200, 200, 200))
            usage_rect = usage_surface.get_rect()
            usage_rect.centerx = int(node.x)
            usage_rect.top = text_rect.bottom + 2
            self.screen.blit(usage_surface, usage_rect)
    
    def _draw_ui(self) -> None:
        """Draw UI elements"""
        if not self.font:
            return
        
        # Draw info panel
        info_lines = [
            f"Nodes: {len(self.nodes)}",
            f"Edges: {len(self.edges)}",
            f"Layout: {self.layout_mode}",
            f"Labels: {'On' if self.show_labels else 'Off'}",
            f"Edges: {'On' if self.show_edges else 'Off'}",
            f"Usage: {'On' if self.show_usage_stats else 'Off'}"
        ]
        
        y_offset = 10
        for line in info_lines:
            text_surface = self.font.render(line, True, self.colors['ui'])
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25
        
        # Draw selected node info
        if self.selected_node:
            self._draw_selected_node_info()
    
    def _draw_selected_node_info(self) -> None:
        """Draw detailed information for selected node"""
        if not self.font or not self.selected_node:
            return
        
        node = self.selected_node
        
        # Create info panel
        info_lines = [
            f"Type: {node.node_type}",
            f"Title: {node.title}",
            f"Usage: {node.usage_count}",
            f"Keywords: {', '.join(node.keywords[:5])}",
        ]
        
        if node.entities:
            info_lines.append(f"Entities: {', '.join(node.entities[:3])}")
        
        if node.timestamp_range:
            info_lines.append(f"Time: {node.timestamp_range}")
        
        # Draw panel background
        panel_width = 300
        panel_height = len(info_lines) * 25 + 20
        panel_rect = pygame.Rect(
            self.screen_width - panel_width - 10,
            self.screen_height - panel_height - 10,
            panel_width, panel_height
        )
        pygame.draw.rect(self.screen, (40, 40, 60), panel_rect)
        pygame.draw.rect(self.screen, (100, 100, 150), panel_rect, 2)
        
        # Draw text
        y_offset = self.screen_height - panel_height
        for line in info_lines:
            text_surface = self.font.render(line, True, self.colors['text'])
            self.screen.blit(text_surface, (self.screen_width - panel_width, y_offset))
            y_offset += 25
    
    def export_graph(self, filename: str = "echogem_graph.json") -> None:
        """Export graph data to JSON file"""
        graph_data = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges)
            }
        }
        
        # Export nodes
        for node in self.nodes.values():
            node_data = {
                "id": node.node_id,
                "type": node.node_type,
                "title": node.title,
                "content": node.content[:200] + "..." if len(node.content) > 200 else node.content,
                "keywords": node.keywords,
                "entities": node.entities,
                "timestamp_range": node.timestamp_range,
                "usage_count": node.usage_count,
                "x": node.x,
                "y": node.y
            }
            graph_data["nodes"].append(node_data)
        
        # Export edges
        for edge in self.edges:
            edge_data = {
                "source": edge.source_id,
                "target": edge.target_id,
                "weight": edge.weight,
                "type": edge.edge_type
            }
            graph_data["edges"].append(edge_data)
        
        # Save to file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            print(f"Graph exported to {filename}")
        except Exception as e:
            print(f"Error exporting graph: {e}")


def main():
    """Main function for running the graph visualizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EchoGem Graph Visualizer")
    parser.add_argument("--usage-cache", default="usage_cache_store.csv", 
                       help="Path to usage cache CSV file")
    parser.add_argument("--width", type=int, default=1200, help="Screen width")
    parser.add_argument("--height", type=int, default=800, help="Screen height")
    parser.add_argument("--export", help="Export graph to JSON file")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = GraphVisualizer(
        usage_cache_path=args.usage_cache,
        screen_width=args.width,
        screen_height=args.height
    )
    
    # Export if requested
    if args.export:
        visualizer.export_graph(args.export)
        return
    
    # Run visualization
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    except Exception as e:
        print(f"Error running visualization: {e}")


if __name__ == "__main__":
    main()
