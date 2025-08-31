# EchoGem Installation Guide

This guide will walk you through installing and setting up EchoGem on your system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Google AI API key
- Pinecone API key

## Step 1: Install Python

Make sure you have Python 3.8+ installed:

```bash
python --version
# or
python3 --version
```

If Python is not installed, download it from [python.org](https://python.org).

## Step 2: Get API Keys

### Google AI API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

### Pinecone API Key
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign up or sign in
3. Go to API Keys section
4. Create a new API key
5. Copy the generated key

## Step 3: Install EchoGem

### Option A: Install from PyPI (Recommended)

```bash
pip install echogem
```

### Option B: Install from Source

```bash
git clone https://github.com/yourusername/echogem.git
cd echogem
pip install -e .
```

## Step 4: Set Environment Variables

### On Linux/macOS:
```bash
export GOOGLE_API_KEY="your-google-api-key-here"
export PINECONE_API_KEY="your-pinecone-api-key-here"
```

### On Windows (Command Prompt):
```cmd
set GOOGLE_API_KEY=your-google-api-key-here
set PINECONE_API_KEY=your-pinecone-api-key-here
```

### On Windows (PowerShell):
```powershell
$env:GOOGLE_API_KEY="your-google-api-key-here"
$env:PINECONE_API_KEY="your-pinecone-api-key-here"
```

### Permanent Setup (Optional)

To make these environment variables permanent:

#### Linux/macOS:
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export GOOGLE_API_KEY="your-key"' >> ~/.bashrc
echo 'export PINECONE_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

#### Windows:
1. Search for "Environment Variables" in Start menu
2. Click "Edit the system environment variables"
3. Click "Environment Variables"
4. Add new variables under "User variables"

## Step 5: Install Optional Dependencies

For better NLP analysis, install the spacy English model:

```bash
python -m spacy download en_core_web_sm
```

## Step 6: Verify Installation

Test that EchoGem is working:

```bash
# Check if the command is available
py -m echogem.cli --help

# Test basic functionality
py -m echogem.cli stats

# Test graph visualization (optional)
py -m echogem.cli graph --export test_graph.json
```

## Step 7: Create Your First Transcript

Create a text file with your transcript:

```bash
echo "This is a sample transcript for testing EchoGem." > transcript.txt
```

## Step 8: Process Your First Transcript

```bash
# Process the transcript
py -m echogem.cli process transcript.txt --show-chunks

# Ask a question
py -m echogem.cli ask "What is this transcript about?"
```

## Troubleshooting

### Common Issues

1. **"Command not found: echogem"**
   - Make sure you installed the package correctly
   - Try `pip list | grep echogem`
   - Restart your terminal after installation

2. **"Missing API keys"**
   - Verify environment variables are set: `echo $GOOGLE_API_KEY`
   - Restart terminal after setting variables
   - Check for typos in the keys

3. **"Pinecone connection error"**
   - Verify your Pinecone API key is correct
   - Check if your Pinecone account is active
   - Ensure you have sufficient quota

4. **"Google API error"**
   - Verify your Google AI API key is correct
   - Check if you have enabled the Gemini API
   - Ensure you have sufficient quota

5. **"Module not found" errors**
   - Try reinstalling: `pip uninstall echogem && pip install echogem`
   - Check Python version compatibility
   - Verify virtual environment if using one

### Getting Help

- Check the [README.md](README.md) for detailed usage information
- Look at the [examples](examples/) directory for code samples
- Open an issue on GitHub if you encounter bugs
- Check the [troubleshooting section](README.md#troubleshooting) in the README

## Next Steps

Once installation is complete:

1. **Read the [README.md](README.md)** for comprehensive usage instructions
2. **Try the examples** in the `examples/` directory
3. **Explore the CLI** with `py -m echogem.cli --help`
4. **Start processing your transcripts** and asking questions!

## Support

If you continue to have issues:

1. Check the [GitHub Issues](https://github.com/yourusername/echogem/issues) page
2. Search for similar problems
3. Create a new issue with detailed error information
4. Include your Python version, OS, and error messages

Happy transcript processing! 🎯
