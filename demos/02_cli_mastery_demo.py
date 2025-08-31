#!/usr/bin/env python3
"""
EchoGem CLI Mastery Demo
=========================

This demo showcases all available CLI commands and their advanced usage.
Learn how to use EchoGem effectively from the command line.

Prerequisites:
- Set GOOGLE_API_KEY environment variable
- Set PINECONE_API_KEY environment variable
- Have sample transcript files ready
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add the parent directory to Python path to import echogem
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables:")
        print("   export GOOGLE_API_KEY='your-google-api-key'")
        print("   export PINECONE_API_KEY='your-pinecone-api-key'")
        return False

    print("✅ Environment variables configured")
    return True

def run_cli_command(command, description, show_output=True):
    """Run a CLI command and display results"""
    print(f"\n🔧 {description}")
    print(f"Command: py -m echogem.cli {command}")
    
    try:
        result = subprocess.run(
            f"py -m echogem.cli {command}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if show_output:
            if result.stdout:
                print("Output:")
                print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            
            if result.stderr:
                print("Errors/Warnings:")
                print(result.stderr[:300] + "..." if len(result.stderr) > 300 else result.stderr)
        
        if result.returncode == 0:
            print("✅ Command executed successfully")
            return True
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def demo_cli_help_commands():
    """Demonstrate all help commands"""
    print("\n" + "="*60)
    print("CLI HELP COMMANDS")
    print("="*60)

    help_commands = [
        ("--help", "General help and command overview"),
        ("process --help", "Help for transcript processing"),
        ("ask --help", "Help for question asking"),
        ("interactive --help", "Help for interactive mode"),
        ("graph --help", "Help for graph visualization"),
        ("stats --help", "Help for statistics"),
        ("clear --help", "Help for data clearing")
    ]

    for command, description in help_commands:
        run_cli_command(command, description, show_output=False)

def demo_process_commands():
    """Demonstrate transcript processing commands"""
    print("\n" + "="*60)
    print("TRANSCRIPT PROCESSING COMMANDS")
    print("="*60)

    # Check if sample transcript exists
    transcript_path = Path("../examples/sample_transcript.txt")
    if not transcript_path.exists():
        print(f"❌ Sample transcript not found: {transcript_path}")
        print("   Please ensure the sample transcript exists before running this demo")
        return

    process_commands = [
        (f"process {transcript_path}", "Basic transcript processing"),
        (f"process {transcript_path} --show-chunks", "Processing with chunk details"),
        (f"process {transcript_path} --chunk-size 800 --overlap 150", "Custom chunking parameters"),
        (f"process {transcript_path} --semantic-chunking", "Explicit semantic chunking"),
        (f"process {transcript_path} --show-chunks --chunk-size 600", "Combined options")
    ]

    for command, description in process_commands:
        success = run_cli_command(command, description)
        if not success:
            print("   Skipping remaining process commands due to failure")
            break
        time.sleep(2)  # Brief pause between commands

def demo_ask_commands():
    """Demonstrate question asking commands"""
    print("\n" + "="*60)
    print("QUESTION ASKING COMMANDS")
    print("="*60)

    ask_commands = [
        ('ask "What is the main topic discussed?"', "Basic question"),
        ('ask "Who are the participants?" --show-chunks', "Question with chunk details"),
        ('ask "What are the key decisions?" --show-chunks --show-metadata', "Question with metadata"),
        ('ask "What is the timeline mentioned?" --max-chunks 8', "Question with more context"),
        ('ask "What security measures are discussed?" --similarity-threshold 0.6', "Question with custom threshold")
    ]

    for command, description in ask_commands:
        success = run_cli_command(command, description)
        if not success:
            print("   Skipping remaining ask commands due to failure")
            break
        time.sleep(2)  # Brief pause between commands

def demo_interactive_mode():
    """Demonstrate interactive mode"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE DEMO")
    print("="*60)

    print("Interactive mode allows you to ask multiple questions without restarting.")
    print("Commands available in interactive mode:")
    print("  - help: Show available commands")
    print("  - stats: Show system statistics")
    print("  - clear: Clear all stored data")
    print("  - chunks: Explore stored chunks")
    print("  - quit: Exit interactive mode")
    
    print("\nTo try interactive mode, run:")
    print("  py -m echogem.cli interactive")
    
    print("\nNote: Interactive mode requires user input, so it's not automated in this demo.")

def demo_graph_commands():
    """Demonstrate graph visualization commands"""
    print("\n" + "="*60)
    print("GRAPH VISUALIZATION COMMANDS")
    print("="*60)

    graph_commands = [
        ("graph", "Launch graph visualization with default settings"),
        ("graph --width 1400 --height 900", "Custom screen dimensions"),
        ("graph --usage-cache usage_cache_store.csv", "Custom cache file"),
        ("graph --export graph_data.json", "Export graph data to JSON")
    ]

    for command, description in graph_commands:
        print(f"\n🔧 {description}")
        print(f"Command: py -m echogem.cli {command}")
        print("Note: Graph commands launch GUI applications and may not work in headless environments")

def demo_utility_commands():
    """Demonstrate utility commands"""
    print("\n" + "="*60)
    print("UTILITY COMMANDS")
    print("="*60)

    utility_commands = [
        ("stats", "Show system statistics and performance metrics"),
        ("clear", "Clear all stored chunks and Q&A pairs")
    ]

    for command, description in utility_commands:
        success = run_cli_command(command, description)
        if not success:
            print("   Skipping remaining utility commands due to failure")
            break
        time.sleep(1)

def demo_advanced_workflows():
    """Demonstrate advanced CLI workflows"""
    print("\n" + "="*60)
    print("ADVANCED CLI WORKFLOWS")
    print("="*60)

    workflows = [
        {
            "name": "Complete Analysis Workflow",
            "description": "Process transcript and ask multiple questions",
            "commands": [
                f"process ../examples/sample_transcript.txt --show-chunks",
                'ask "What is the main topic?" --show-chunks',
                'ask "Who are the key speakers?" --show-metadata',
                'ask "What are the action items?" --show-chunks',
                "stats"
            ]
        },
        {
            "name": "Performance Testing Workflow",
            "description": "Test system performance with different parameters",
            "commands": [
                f"process ../examples/sample_transcript.txt --chunk-size 500 --overlap 100",
                'ask "What is the main topic?" --max-chunks 3',
                'ask "What are the key points?" --max-chunks 5',
                'ask "What decisions were made?" --max-chunks 7',
                "stats"
            ]
        }
    ]

    for workflow in workflows:
        print(f"\n🔄 {workflow['name']}")
        print(f"Purpose: {workflow['description']}")
        print("Commands to run:")
        
        for i, command in enumerate(workflow['commands'], 1):
            print(f"  {i}. py -m echogem.cli {command}")
        
        print("\nTo execute this workflow, run the commands in sequence.")

def demo_error_handling():
    """Demonstrate error handling in CLI"""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMO")
    print("="*60)

    error_scenarios = [
        ("process nonexistent_file.txt", "File not found error"),
        ('ask "What is this about?"', "Question without processed data"),
        ("stats", "Statistics with no data"),
        ("process ../examples/sample_transcript.txt --chunk-size -100", "Invalid parameter")
    ]

    for command, description in error_scenarios:
        print(f"\n🔧 {description}")
        print(f"Command: py -m echogem.cli {command}")
        print("Expected: Error message with helpful information")
        print("Note: This demonstrates how EchoGem handles errors gracefully")

def demo_cli_tips():
    """Share CLI usage tips and best practices"""
    print("\n" + "="*60)
    print("CLI USAGE TIPS & BEST PRACTICES")
    print("="*60)

    tips = [
        "Always use 'py -m echogem.cli' instead of just 'echogem'",
        "Use --show-chunks to understand how your data is being processed",
        "Combine --show-chunks and --show-metadata for comprehensive analysis",
        "Use --max-chunks to control context window size for questions",
        "Adjust --similarity-threshold based on your content type",
        "Use --chunk-size and --overlap to optimize for your document type",
        "Run 'stats' regularly to monitor system performance",
        "Use 'clear' to reset the system when starting fresh",
        "Interactive mode is great for exploration sessions",
        "Graph visualization helps understand information relationships"
    ]

    for i, tip in enumerate(tips, 1):
        print(f"{i:2d}. {tip}")

def main():
    """Main demo function"""
    print("🚀 EchoGem CLI Mastery Demo")
    print("="*60)

    # Check environment
    if not check_environment():
        print("\n⚠️  Please set the required environment variables first.")
        print("   You can still view the CLI commands and examples below.")

    # Run all CLI demonstrations
    demo_cli_help_commands()
    demo_process_commands()
    demo_ask_commands()
    demo_interactive_mode()
    demo_graph_commands()
    demo_utility_commands()
    demo_advanced_workflows()
    demo_error_handling()
    demo_cli_tips()

    print("\n" + "="*60)
    print("CLI MASTERY COMPLETE!")
    print("="*60)
    
    print("\n🎯 What you've learned:")
    print("   ✓ All available CLI commands and options")
    print("   ✓ Advanced parameter combinations")
    print("   ✓ Error handling and troubleshooting")
    print("   ✓ Best practices and optimization tips")
    print("   ✓ Complete workflow examples")

    print("\n💡 Next steps:")
    print("   1. Practice with your own transcript files")
    print("   2. Experiment with different parameter combinations")
    print("   3. Try the interactive mode for exploration")
    print("   4. Use graph visualization to understand data relationships")
    print("   5. Check out other demo files for more examples")

    print("\n📚 For more information:")
    print("   - CLI Guide: ../docs/CLI_GUIDE.md")
    print("   - User Guide: ../docs/USER_GUIDE.md")
    print("   - API Reference: ../docs/API_REFERENCE.md")

if __name__ == "__main__":
    main()
