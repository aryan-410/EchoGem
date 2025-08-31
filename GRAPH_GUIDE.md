# EchoGem Graph Visualization Guide

## 🎯 What is the Graph Visualization?

The EchoGem Graph Visualization is an interactive GUI that shows your transcript chunks and Q&A pairs as connected nodes, helping you understand:

- **Information Flow**: How different chunks relate to each other
- **Usage Patterns**: Which chunks are used most frequently
- **Knowledge Structure**: The organization of your transcript data
- **Query Relevance**: Which chunks are most relevant to specific questions

## 🚀 Quick Start

### 1. Install EchoGem
```bash
pip install -e .
```

### 2. Process a Transcript
```bash
py -m echogem.cli process your_transcript.txt
```

### 3. Ask Some Questions
```bash
py -m echogem.cli ask "What is the main topic?"
py -m echogem.cli ask "Who are the key speakers?"
```

### 4. Launch the Graph Visualization
```bash
py -m echogem.cli graph
```

## 🎮 Interactive Controls

### Mouse Controls
- **Left Click**: Select nodes
- **Drag**: Move nodes around
- **Hover**: See node details

### Keyboard Controls
- **SPACE**: Cycle through layout modes
- **L**: Toggle node labels
- **E**: Toggle edge display
- **U**: Toggle usage statistics
- **ESC**: Exit visualization

### Layout Modes
- **Force-directed**: Natural physics-based layout
- **Circular**: Organized circular arrangement
- **Hierarchical**: Tree-like structure

## 📊 Understanding the Visualization

### Node Types
- **🔵 Blue Nodes**: Transcript chunks
- **🟢 Green Nodes**: Q&A pairs

### Edge Types
- **Gray Edges**: Similarity between chunks
- **Orange Edges**: Usage relationships

### Node Size
- Larger nodes = More frequently used
- Smaller nodes = Less frequently used

## 🎨 Customization Options

### Window Size
```bash
py -m echogem.cli graph --width 1600 --height 1000
```

### Export Data
```bash
py -m echogem.cli graph --export my_analysis.json
```

### Custom Usage Cache
```bash
py -m echogem.cli graph --usage-cache custom_cache.csv
```

## 💡 Use Cases

### For Researchers
- Visualize research interview data
- Identify key themes and connections
- Track information usage patterns

### For Students
- Understand lecture content structure
- Review study material relationships
- Prepare for exams with visual aids

### For Professionals
- Analyze meeting transcripts
- Track project discussion threads
- Share insights with team members

### For Content Creators
- Organize podcast or video content
- Identify recurring topics
- Plan future content based on usage

## 🔍 Troubleshooting

### No Nodes Appear
- Make sure you've processed a transcript
- Check that you've made some queries
- Verify the usage cache file exists

### Window Won't Open
- Ensure pygame is installed: `pip install pygame`
- Check your display settings
- Try different window dimensions

### Performance Issues
- Reduce window size for large datasets
- Use export feature for static analysis
- Close other applications to free memory

## 📚 Advanced Features

### Data Export
The exported JSON contains:
- Node information (chunks, Q&A pairs)
- Edge relationships
- Usage statistics
- Timestamps and metadata

### Integration
- Import exported data into other tools
- Use for presentations and reports
- Share with colleagues

## 🆘 Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure you have processed data to visualize
4. Try the demo script: `python examples/graph_visualization_demo.py`

---

**Happy Visualizing! 🎉**

The graph visualization makes your transcript data come alive, helping you discover insights that might be hidden in plain text.
