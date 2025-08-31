#!/usr/bin/env python3
"""
Command-line interface for EchoGem library.
"""

import argparse
import os
import sys
from typing import Optional
from .processor import Processor


def setup_environment():
    """Check and setup required environment variables"""
    required_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables:")
        print("   export GOOGLE_API_KEY='your-google-api-key'")
        print("   export PINECONE_API_KEY='your-pinecone-api-key'")
        return False
    
    print("Environment variables configured")
    return True


def process_transcript(processor: Processor, file_path: str, show_chunks: bool = False):
    """Process a transcript file"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    try:
        processor.chunk_and_process(file_path, output_chunks=show_chunks)
        print(f"Successfully processed transcript: {file_path}")
        return True
    except Exception as e:
        print(f"Error processing transcript: {e}")
        return False


def ask_question(processor: Processor, question: str, show_chunks: bool = False, show_metadata: bool = False, show_pa_pairs: bool = False):
    """Ask a question and get an answer"""
    try:
        result = processor.answer_question(
            question, 
            show_chunks=show_chunks, 
            show_metadata=show_metadata,
            show_pa_pairs=show_pa_pairs
        )
        
        print(f"\nANSWER:")
        print(f"{result.answer}")
        
        if result.chunks_used:
            print(f"\nUsed {len(result.chunks_used)} chunks (confidence: {result.confidence:.2f})")
        
        return True
    except Exception as e:
        print(f"Error answering question: {e}")
        return False


def interactive_mode(processor: Processor):
    """Run in interactive mode for asking multiple questions"""
    print("\nInteractive Mode - Ask questions about your transcript!")
    print("Type 'quit' or 'exit' to end, 'help' for commands")
    
    # Display settings
    show_chunks = False
    show_metadata = False
    show_pa_pairs = False
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif question.lower() in ['help', 'h']:
                print("\nAvailable commands:")
                print("  help, h     - Show this help")
                print("  quit, exit, q - Exit interactive mode")
                print("  stats       - Show system statistics")
                print("  clear       - Clear all data")
                print("  clear-cache - Clear only usage cache")
                print("  chunks <k>  - Show top k chunks for a question")
                print("  show chunks - Toggle chunk display")
                print("  show metadata - Toggle metadata display")
                print("  show pa-pairs - Toggle prompt-answer pairs display")
                print("  merge <id1> <id2> - Merge two chunks")
                print("  radius <id> - Calculate chunk clustering radius")
                print("  Any other text will be treated as a question")
                continue
            elif question.lower() == 'stats':
                stats = processor.get_stats()
                print(f"\nSystem Statistics:")
                print(f"  Usage cache size: {stats.get('usage_cache_size', 'N/A')}")
                print(f"  Chunk index: {stats.get('chunks', 'N/A')}")
                print(f"  PA index: {stats.get('prompt_answers', 'N/A')}")
                continue
            elif question.lower() == 'clear':
                confirm = input("This will delete ALL data. Are you sure? (yes/no): ")
                if confirm.lower() == 'yes':
                    processor.clear_all_data()
                    print("All data cleared")
                continue
            elif question.lower() == 'clear-cache':
                confirm = input("This will clear only the usage cache. Are you sure? (yes/no): ")
                if confirm.lower() == 'yes':
                    processor.clear_cache()
                    print("Usage cache cleared")
                continue
            elif question.lower().startswith('chunks '):
                try:
                    k = int(question.split()[1])
                    test_question = input("Enter a question to find relevant chunks: ").strip()
                    chunks = processor.pick_chunks(test_question, k=k)
                    if chunks:
                        print(f"\nTop {len(chunks)} relevant chunks:")
                        for i, chunk in enumerate(chunks, 1):
                            print(f"\n{i}. {chunk.title}")
                            print(f"   Content: {chunk.content[:150]}...")
                            print(f"   Keywords: {chunk.keywords}")
                    else:
                        print("No chunks found")
                except (ValueError, IndexError):
                    print("Usage: chunks <number>")
                continue
            elif question.lower() == 'show chunks':
                show_chunks = not show_chunks
                print(f"Chunk display: {'ON' if show_chunks else 'OFF'}")
                continue
            elif question.lower() == 'show metadata':
                show_metadata = not show_metadata
                print(f"Metadata display: {'ON' if show_metadata else 'OFF'}")
                continue
            elif question.lower() == 'show pa-pairs':
                show_pa_pairs = not show_pa_pairs
                print(f"Prompt-answer pairs display: {'ON' if show_pa_pairs else 'OFF'}")
                continue
            elif question.lower().startswith('merge '):
                try:
                    parts = question.split()
                    if len(parts) != 3:
                        print("Usage: merge <chunk_id1> <chunk_id2>")
                        continue
                    
                    chunk_id1, chunk_id2 = parts[1], parts[2]
                    
                    # Get chunks from vector store (this would need to be implemented)
                    print(f"Attempting to merge chunks: {chunk_id1} + {chunk_id2}")
                    print("Note: This feature requires chunk retrieval from vector store")
                    print("   Implementation needed: get_chunk_by_id() method")
                    
                except Exception as e:
                    print(f"Error merging chunks: {e}")
                continue
            elif question.lower().startswith('radius '):
                try:
                    parts = question.split()
                    if len(parts) != 2:
                        print("Usage: radius <chunk_id>")
                        continue
                    
                    chunk_id = parts[1]
                    
                    # Get chunk from vector store (this would need to be implemented)
                    print(f"Calculating radius for chunk: {chunk_id}")
                    print("Note: This feature requires chunk retrieval from vector store")
                    print("   Implementation needed: get_chunk_by_id() method")
                    
                except Exception as e:
                    print(f"Error calculating radius: {e}")
                continue
            elif not question:
                continue
            
            # Process the question
            ask_question(processor, question, show_chunks=show_chunks, show_metadata=show_metadata, show_pa_pairs=show_pa_pairs)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="EchoGem - Intelligent Transcript Processing and Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a transcript file
  py -m echogem.cli process transcript.txt
  
  # Process and show chunk details
  py -m echogem.cli process transcript.txt --show-chunks
  
  # Ask a single question
  py -m echogem.cli ask "What is the main topic discussed?"
  
     # Ask with chunk details
   py -m echogem.cli ask "What is the main topic discussed?" --show-chunks --show-metadata --show-pa-pairs
  
  # Interactive mode
  py -m echogem.cli interactive
  
  # Show system statistics
  py -m echogem.cli stats
  
  # Clear all data
  py -m echogem.cli clear
  
  # Visualize information graph
  py -m echogem.cli graph
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a transcript file')
    process_parser.add_argument('file', help='Path to transcript file')
    process_parser.add_argument('--show-chunks', action='store_true', help='Show chunk details')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument('--show-chunks', action='store_true', help='Show retrieved chunks')
    ask_parser.add_argument('--show-metadata', action='store_true', help='Show chunk metadata')
    ask_parser.add_argument('--show-pa-pairs', action='store_true', help='Show similar prompt-answer pairs')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Run in interactive mode')
    
    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')
    
    # Clear command
    subparsers.add_parser('clear', help='Clear all stored data')
    
    # Graph command
    graph_parser = subparsers.add_parser('graph', help='Visualize information graph')
    graph_parser.add_argument('--usage-cache', default="usage_cache_store.csv", 
                             help='Path to usage cache CSV file')
    graph_parser.add_argument('--width', type=int, default=1200, help='Screen width')
    graph_parser.add_argument('--height', type=int, default=800, help='Screen height')
    graph_parser.add_argument('--export', help='Export graph to JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check environment (except for graph command)
    if args.command != 'graph' and not setup_environment():
        sys.exit(1)
    
    try:
        if args.command == 'graph':
            # Import and run graph visualizer
            from .graphe import GraphVisualizer
            
            visualizer = GraphVisualizer(
                usage_cache_path=args.usage_cache,
                screen_width=args.width,
                screen_height=args.height
            )
            
            if args.export:
                visualizer.export_graph(args.export)
            else:
                visualizer.run()
                
        else:
            # Initialize processor for other commands
            processor = Processor()
            
            if args.command == 'process':
                success = process_transcript(processor, args.file, args.show_chunks)
                sys.exit(0 if success else 1)
                
            elif args.command == 'ask':
                success = ask_question(processor, args.question, args.show_chunks, args.show_metadata, args.show_pa_pairs)
                sys.exit(0 if success else 1)
                
            elif args.command == 'clear':
                confirm = input("This will delete ALL data. Are you sure? (yes/no): ")
                if confirm.lower() == 'yes':
                    processor.clear_all_data()
                    print("All data cleared")
                else:
                    print("Operation cancelled")
                sys.exit(0)
                
            elif args.command == 'interactive':
                interactive_mode(processor)
                
            elif args.command == 'stats':
                stats = processor.get_stats()
                print(f"\nSystem Statistics:")
                print(f"  Usage cache size: {stats.get('usage_cache_size', 'N/A')}")
                print(f"  Chunk index: {stats.get('chunks', 'N/A')}")
                print(f"  PA index: {stats.get('prompt_answers', 'N/A')}")
                
            elif args.command == 'clear':
                confirm = input("This will delete ALL data. Are you sure? (yes/no): ")
                if confirm.lower() == 'yes':
                    processor.clear_all_data()
                    print("All data cleared")
                else:
                    print("Operation cancelled")
                    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
