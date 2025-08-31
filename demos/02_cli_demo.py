#!/usr/bin/env python3
"""
EchoGem CLI Demo

This demo showcases all the command-line interface features:
- Processing transcripts
- Querying the knowledge base
- Graph visualization
- Usage statistics
- Export functionality
"""

import os
import subprocess
import time
import json
from pathlib import Path
import tempfile

def create_sample_files():
    """Create sample transcript files for CLI testing"""
    samples = {
        "tech_interview.txt": """
        Interviewer: Welcome to our technical interview. Can you tell us about your experience with machine learning?
        
        Candidate: Thank you for having me. I've been working with machine learning for about three years now. I started with scikit-learn for basic classification tasks, then moved to TensorFlow for deep learning projects.
        
        Interviewer: That's great. Can you walk us through a specific project where you used machine learning?
        
        Candidate: Absolutely. I worked on a customer churn prediction model for an e-commerce company. We had about 100,000 customer records with features like purchase frequency, average order value, and time since last purchase.
        
        I used a random forest classifier initially, which gave us about 78% accuracy. Then I implemented a neural network with dropout layers, which improved accuracy to 82%. The key insight was that customers who hadn't made a purchase in the last 30 days were 3x more likely to churn.
        
        Interviewer: How did you handle the imbalanced dataset?
        
        Candidate: Good question. The churn rate was only about 15%, so I used SMOTE to generate synthetic minority samples. I also adjusted the class weights in the loss function to give more importance to the minority class.
        
        Interviewer: What metrics did you use to evaluate the model?
        
        Candidate: I focused on precision and recall rather than just accuracy, since false positives and false negatives have different business implications. I also used ROC-AUC and precision-recall curves to get a complete picture of performance.
        """,
        
        "meeting_notes.txt": """
        Team Meeting - Q4 Planning Session
        Date: December 15, 2024
        Attendees: Sarah (PM), Mike (Dev), Lisa (Design), Tom (QA)
        
        Sarah: Let's review our Q4 goals and plan the next sprint. We have three major features to deliver: user authentication, payment integration, and the new dashboard.
        
        Mike: For user authentication, I've completed the backend API. We're using JWT tokens with refresh token rotation. The frontend components are about 70% done.
        
        Lisa: I've finished the design mockups for the payment integration. We're going with Stripe for processing and a clean, minimal interface that matches our brand guidelines.
        
        Tom: I've started writing test cases for the authentication flow. We should have full coverage by the end of the week.
        
        Sarah: Great progress. What are our blockers?
        
        Mike: The payment gateway integration is taking longer than expected due to Stripe's new security requirements. We might need an extra week.
        
        Lisa: No blockers on the design side. All assets are ready for development.
        
        Tom: We need to set up the CI/CD pipeline for automated testing. Currently, tests are running manually.
        
        Sarah: Let's prioritize the authentication feature for the next sprint, then tackle payments. Tom, can you work with Mike to set up the CI/CD pipeline?
        
        Tom: Absolutely. I'll start with GitHub Actions and we can migrate to Jenkins later if needed.
        
        Next Actions:
        - Mike: Complete frontend authentication components
        - Lisa: Finalize payment design specs
        - Tom: Set up CI/CD pipeline
        - Sarah: Update project timeline
        """,
        
        "research_paper.txt": """
        Abstract: This paper presents a novel approach to sentiment analysis using transformer-based models and domain adaptation techniques. We evaluate our method on three benchmark datasets and achieve state-of-the-art performance.
        
        Introduction: Sentiment analysis has become increasingly important in natural language processing, with applications ranging from social media monitoring to customer feedback analysis. Traditional approaches rely on hand-crafted features and shallow learning models, which often fail to capture the nuanced semantics of human language.
        
        Related Work: Previous research has explored various approaches to sentiment analysis. Pang et al. (2002) used support vector machines with n-gram features. More recently, deep learning approaches have shown promising results, particularly with recurrent neural networks and attention mechanisms.
        
        Methodology: Our approach combines BERT embeddings with domain-specific fine-tuning. We introduce a novel loss function that incorporates domain knowledge through adversarial training. The model architecture consists of a BERT encoder followed by a classification head with dropout regularization.
        
        Experiments: We evaluate our model on the Stanford Sentiment Treebank, IMDB movie reviews, and Amazon product reviews. Our method achieves 94.2% accuracy on SST, 92.8% on IMDB, and 89.5% on Amazon, outperforming previous state-of-the-art results by 2.1%, 1.8%, and 2.5% respectively.
        
        Results: The domain adaptation component significantly improves performance on out-of-domain data. Ablation studies show that both the adversarial training and the novel loss function contribute to the performance gains.
        
        Conclusion: We have presented a novel approach to sentiment analysis that leverages transformer models and domain adaptation. Our results demonstrate the effectiveness of this approach across multiple domains and datasets.
        """
    }
    
    files_created = []
    for filename, content in samples.items():
        filepath = Path(filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        files_created.append(filepath)
        print(f"   📝 Created: {filename}")
    
    return files_created

def run_cli_command(command, description, capture_output=True):
    """Run a CLI command and display results"""
    print(f"\n🔧 {description}")
    print("-" * 50)
    print(f"Command: {command}")
    print()
    
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            if result.stdout:
                print("📤 Output:")
                print(result.stdout)
            if result.stderr:
                print("⚠️  Errors/Warnings:")
                print(result.stderr)
            print(f"Exit code: {result.returncode}")
            return result.returncode == 0
        else:
            # For interactive commands, don't capture output
            result = subprocess.run(command, shell=True, timeout=60)
            return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False

def demo_cli_features():
    """Demonstrate all CLI features"""
    print("🎮 EchoGem CLI Demo")
    print("=" * 40)
    print("This demo will show you how to use EchoGem from the command line!")
    print()
    
    # Check if echogem is installed
    print("1️⃣ Checking EchoGem installation...")
    try:
        result = subprocess.run("echogem --help", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ EchoGem CLI is available")
        else:
            print("   ❌ EchoGem CLI not found")
            print("   💡 Install with: pip install -e .")
            return False
    except Exception as e:
        print(f"   ❌ Error checking installation: {e}")
        return False
    
    # Create sample files
    print("\n2️⃣ Creating sample transcript files...")
    sample_files = create_sample_files()
    
    # Process transcripts
    print("\n3️⃣ Processing transcripts with CLI...")
    for filepath in sample_files:
        success = run_cli_command(
            f"echogem process {filepath} --chunk-size 300 --overlap 50",
            f"Processing {filepath.name}"
        )
        if not success:
            print(f"   ⚠️  Processing {filepath.name} failed, continuing...")
    
    # Query the knowledge base
    print("\n4️⃣ Querying the knowledge base...")
    questions = [
        "What are the main topics discussed in the technical interview?",
        "What are the Q4 goals mentioned in the meeting?",
        "What is the methodology used in the research paper?",
        "How does the sentiment analysis model perform?",
        "What are the blockers mentioned in the meeting?"
    ]
    
    for question in questions:
        success = run_cli_command(
            f'echogem query "{question}" --show-chunks --show-prompt-answers --max-chunks 2',
            f"Query: {question}"
        )
        if not success:
            print(f"   ⚠️  Query failed, continuing...")
    
    # Show usage statistics
    print("\n5️⃣ Usage statistics...")
    run_cli_command(
        "echogem stats",
        "Displaying usage statistics"
    )
    
    # Export functionality
    print("\n6️⃣ Export functionality...")
    run_cli_command(
        "echogem export chunks --output exported_chunks.json",
        "Exporting chunks to JSON"
    )
    
    run_cli_command(
        "echogem export qa --output exported_qa.json",
        "Exporting Q&A pairs to JSON"
    )
    
    # Graph visualization
    print("\n7️⃣ Graph visualization...")
    print("   🎨 Launching graph visualization (this will open a window)")
    print("   💡 Use ESC to exit when done exploring")
    
    # Run graph command in background (non-blocking)
    try:
        process = subprocess.Popen("echogem graph --width 1000 --height 700", shell=True)
        print("   🚀 Graph visualization launched!")
        print("   💡 Press Enter when you're done exploring the graph...")
        input()
        process.terminate()
        print("   ✅ Graph visualization closed")
    except Exception as e:
        print(f"   ❌ Graph visualization failed: {e}")
    
    return True

def demo_advanced_cli():
    """Demonstrate advanced CLI features"""
    print("\n8️⃣ Advanced CLI Features")
    print("=" * 35)
    
    # Batch processing
    print("\n📦 Batch processing multiple files...")
    run_cli_command(
        "echogem batch-process *.txt --chunk-size 400 --overlap 75",
        "Batch processing all text files"
    )
    
    # Custom chunking options
    print("\n⚙️  Custom chunking options...")
    run_cli_command(
        "echogem process tech_interview.txt --chunk-size 200 --overlap 25 --semantic-chunking",
        "Processing with custom chunking parameters"
    )
    
    # Query with specific options
    print("\n🔍 Advanced querying...")
    run_cli_command(
        'echogem query "machine learning experience" --max-chunks 5 --similarity-threshold 0.7',
        "Query with advanced parameters"
    )
    
    # Export with filtering
    print("\n📤 Export with filtering...")
    run_cli_command(
        "echogem export chunks --output recent_chunks.json --filter recent --limit 10",
        "Exporting recent chunks with filtering"
    )

def cleanup_sample_files():
    """Clean up sample files created during demo"""
    print("\n🧹 Cleaning up sample files...")
    sample_files = ["tech_interview.txt", "meeting_notes.txt", "research_paper.txt"]
    
    for filename in sample_files:
        try:
            if Path(filename).exists():
                Path(filename).unlink()
                print(f"   🗑️  Deleted: {filename}")
        except Exception as e:
            print(f"   ⚠️  Could not delete {filename}: {e}")

def main():
    """Main CLI demo function"""
    print("🎯 EchoGem Command-Line Interface Demonstration")
    print("=" * 65)
    print("This demo will show you how to use EchoGem from the command line!")
    print()
    
    # Run basic CLI demo
    success = demo_cli_features()
    if not success:
        print("\n❌ CLI demo failed. Please check your setup and try again.")
        return
    
    # Run advanced features
    demo_advanced_cli()
    
    # Cleanup
    cleanup_sample_files()
    
    # Final recommendations
    print("\n🎉 CLI Demo Complete!")
    print("=" * 25)
    print("💡 Key CLI commands to remember:")
    print("   📝 Process: echogem process <file> [options]")
    print("   ❓ Query: echogem query <question> [options]")
    print("   📊 Stats: echogem stats")
    print("   🎨 Graph: echogem graph [options]")
    print("   📤 Export: echogem export <type> [options]")
    print("   🔧 Help: echogem --help")
    
    print("\n📚 Explore other demos:")
    print("   - Basic workflow: python demos/01_basic_workflow_demo.py")
    print("   - Python API: python demos/03_api_demo.py")
    print("   - Performance: python demos/09_performance_benchmarking_demo.py")

if __name__ == "__main__":
    main()
