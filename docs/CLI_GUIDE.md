# EchoGem CLI Usage Guide

## 🚀 Quick Start

The EchoGem library provides a powerful command-line interface for processing transcripts and asking questions. **Important**: Use the correct command format:

```bash
# ❌ WRONG - This won't work:
echogem process transcript.txt

# ✅ CORRECT - Use the CLI module:
py -m echogem.cli process transcript.txt
```

## 📋 Available Commands

### 1. Process Transcripts

```bash
# Basic processing
py -m echogem.cli process transcript.txt

# Show chunk details during processing
py -m echogem.cli process transcript.txt --show-chunks

# Process with custom options
py -m echogem.cli process transcript.txt --chunk-size 1000 --overlap 200
```

### 2. Ask Questions

```bash
# Basic question
py -m echogem.cli ask "What is the main topic discussed?"

# Show which chunks were used
py -m echogem.cli ask "What are the key findings?" --show-chunks

# Show chunk metadata
py -m echogem.cli ask "Who are the speakers?" --show-chunks --show-metadata

# Use more context
py -m echogem.cli ask "Explain the methodology" --max-chunks 10
```

### 3. Interactive Mode

```bash
# Launch interactive Q&A session
py -m echogem.cli interactive
```

In interactive mode, you can:
- Ask multiple questions without restarting
- View system statistics
- Clear data
- Explore chunks
- Type 'help' for available commands
- Type 'quit' to exit

### 4. Graph Visualization

```bash
# Launch interactive graph
py -m echogem.cli graph

# Custom screen size
py -m echogem.cli graph --width 1400 --height 900

# Export graph data
py -m echogem.cli graph --export graph_data.json
```

### 5. System Management

```bash
# Show system statistics
py -m echogem.cli stats

# Clear all stored data
py -m echogem.cli clear
```

## 🔧 Command Options

### Process Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--show-chunks` | Display chunk details during processing | False |
| `--chunk-size` | Maximum chunk size in characters | 1000 |
| `--overlap` | Overlap between consecutive chunks | 200 |
| `--semantic-chunking` | Use AI-powered semantic chunking | True |

### Ask Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--show-chunks` | Show which chunks were retrieved | False |
| `--show-metadata` | Display chunk metadata | False |
| `--max-chunks` | Maximum number of chunks to use | 5 |
| `--similarity-threshold` | Minimum similarity score | 0.7 |

### Graph Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--width` | Screen width in pixels | 1200 |
| `--height` | Screen height in pixels | 800 |
| `--usage-cache` | Path to usage cache file | usage_cache_store.csv |
| `--export` | Export graph data to JSON file | None |

## 📚 Complete Workflow Example

### Step 1: Set Environment Variables

```bash
# PowerShell
$env:GOOGLE_API_KEY="your-google-api-key-here"
$env:PINECONE_API_KEY="your-pinecone-api-key-here"

# Bash/Linux/macOS
export GOOGLE_API_KEY="your-google-api-key-here"
export PINECONE_API_KEY="your-pinecone-api-key-here"
```

### Step 2: Process Your Transcript

```bash
py -m echogem.cli process meeting_transcript.txt --show-chunks
```

### Step 3: Ask Questions

```bash
# Basic question
py -m echogem.cli ask "What is the main topic discussed?"

# Detailed analysis
py -m echogem.cli ask "What are the key decisions made?" --show-chunks --show-metadata
```

### Step 4: Interactive Exploration

```bash
py -m echogem.cli interactive
```

### Step 5: Visualize Information Flow

```bash
py -m echogem.cli graph
```

## 🎯 Common Use Cases

### Academic Paper Analysis

```bash
# Process research paper
py -m echogem.cli process research_paper.txt --show-chunks

# Ask about methodology
py -m echogem.cli ask "What methodology was used in this research?" --show-chunks

# Ask about findings
py -m echogem.cli ask "What are the main findings and conclusions?" --show-chunks --show-metadata
```

### Meeting Transcript Analysis

```bash
# Process meeting recording
py -m echogem.cli process team_meeting.txt

# Find action items
py -m echogem.cli ask "What action items were assigned and to whom?" --show-chunks

# Understand decisions
py -m echogem.cli ask "What key decisions were made in this meeting?" --show-chunks
```

### Interview Analysis

```bash
# Process interview transcript
py -m echogem.cli process candidate_interview.txt

# Evaluate technical skills
py -m echogem.cli ask "What technical skills and experience does the candidate have?" --show-chunks

# Assess communication
py -m echogem.cli ask "How well did the candidate communicate their ideas?" --show-chunks
```

## 🚨 Troubleshooting

### Common Issues

1. **"No module named echogem"**
   - Use `py -m echogem.cli` instead of `echogem`
   - Ensure the package is installed: `pip install echogem`

2. **"Missing environment variables"**
   - Set `GOOGLE_API_KEY` and `PINECONE_API_KEY`
   - Use PowerShell: `$env:VARIABLE_NAME="value"`
   - Use Bash: `export VARIABLE_NAME="value"`

3. **"File not found"**
   - Check file path and ensure file exists
   - Use absolute paths if needed

4. **"API key invalid"**
   - Verify your Google Gemini API key
   - Check Pinecone API key and environment

### Getting Help

```bash
# Show general help
py -m echogem.cli --help

# Show help for specific command
py -m echogem.cli process --help
py -m echogem.cli ask --help
py -m echogem.cli graph --help
```

## 🔗 Related Documentation

- [User Guide](USER_GUIDE.md) - Complete user documentation
- [API Reference](API_REFERENCE.md) - Programmatic usage
- [Architecture](ARCHITECTURE.md) - System design details
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions

## 💡 Tips and Best Practices

1. **Start Simple**: Begin with basic processing and questions
2. **Use Show Chunks**: Enable `--show-chunks` to understand how the system works
3. **Interactive Mode**: Use interactive mode for exploration sessions
4. **Graph Visualization**: Use the graph to understand information relationships
5. **Environment Variables**: Set API keys once per session
6. **File Formats**: Support for .txt, .md, and other text formats
7. **Chunking Options**: Adjust chunk size and overlap for your content type
