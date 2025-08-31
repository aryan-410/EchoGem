#!/usr/bin/env python3
"""
EchoGem Academic Paper Demo

This demo showcases processing and analyzing academic research papers:
- Scientific paper structure analysis
- Citation and reference extraction
- Methodology understanding
- Results and conclusion analysis
- Cross-paper relationship discovery
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from processor import Processor
from models import ChunkingOptions, QueryOptions

def create_academic_papers():
    """Create sample academic papers for demonstration"""
    papers = {
        "quantum_machine_learning": """
        Title: Quantum Machine Learning: A Survey of Current Approaches and Future Directions
        Authors: Dr. Sarah Chen, Dr. Michael Rodriguez, Dr. Emily Watson
        Institution: Stanford University, Department of Computer Science
        Publication Date: March 2024
        Journal: Nature Machine Intelligence
        DOI: 10.1038/s42256-024-00001-x
        
        Abstract:
        Quantum machine learning represents a paradigm shift in computational approaches to artificial intelligence. This survey provides a comprehensive overview of current quantum machine learning algorithms, their theoretical foundations, and practical applications. We examine quantum versions of classical machine learning techniques including quantum neural networks, quantum support vector machines, and quantum clustering algorithms. Our analysis reveals that quantum machine learning offers exponential speedups for specific problem classes while maintaining compatibility with classical machine learning frameworks.
        
        Introduction:
        The intersection of quantum computing and machine learning has emerged as one of the most promising frontiers in computational science. Classical machine learning algorithms face fundamental limitations when dealing with high-dimensional data and complex optimization problems. Quantum computing, with its inherent parallelism and superposition properties, offers novel approaches to these challenges.
        
        Recent advances in quantum hardware have made it possible to implement quantum machine learning algorithms on real quantum devices. IBM's 433-qubit Osprey processor and Google's 53-qubit Sycamore processor have demonstrated quantum advantage in specific computational tasks. These developments have accelerated research in quantum machine learning algorithms and their applications.
        
        Related Work:
        Previous surveys have focused on specific aspects of quantum machine learning. Schuld and Petruccione (2018) provided an introduction to quantum machine learning concepts, while Biamonte et al. (2017) focused on quantum algorithms for machine learning tasks. Our work extends these surveys by incorporating recent developments in quantum hardware and algorithm design.
        
        Several research groups have developed quantum versions of classical machine learning algorithms. Havlíček et al. (2019) demonstrated quantum advantage in binary classification tasks using quantum feature maps. Schuld et al. (2020) developed quantum neural networks with trainable quantum circuits. These approaches form the foundation for more sophisticated quantum machine learning systems.
        
        Methodology:
        Our survey methodology involves systematic analysis of peer-reviewed publications in quantum machine learning from 2018 to 2024. We identified 247 relevant papers through keyword searches in major scientific databases including arXiv, IEEE Xplore, and ACM Digital Library. Papers were categorized based on their approach to quantum machine learning: quantum-inspired algorithms, hybrid quantum-classical methods, and fully quantum algorithms.
        
        For each category, we analyzed algorithmic complexity, theoretical guarantees, and experimental validation. We also examined the relationship between quantum hardware capabilities and algorithm performance. This analysis revealed important insights into the current state and future potential of quantum machine learning.
        
        Quantum Neural Networks:
        Quantum neural networks represent one of the most active areas of research in quantum machine learning. These networks use quantum circuits to perform neural network operations, potentially offering exponential speedups for certain computational tasks. The key innovation in quantum neural networks is the use of parameterized quantum gates that can be optimized through classical optimization techniques.
        
        Several architectures have been proposed for quantum neural networks. The quantum circuit learning framework introduced by Mitarai et al. (2018) uses parameterized quantum gates to create trainable quantum circuits. The quantum approximate optimization algorithm (QAOA) developed by Farhi et al. (2014) can be viewed as a quantum neural network for optimization problems.
        
        Experimental results with quantum neural networks have shown promising results. McClean et al. (2018) demonstrated that quantum neural networks can learn complex functions with fewer parameters than classical neural networks. However, challenges remain in training quantum neural networks due to the barren plateau problem, where gradients become exponentially small as the number of qubits increases.
        
        Quantum Feature Maps:
        Quantum feature maps provide a bridge between classical and quantum machine learning by encoding classical data into quantum states. This approach allows classical machine learning algorithms to benefit from quantum computational advantages without requiring fully quantum algorithms. Quantum feature maps can be implemented using various quantum encoding strategies including amplitude encoding, angle encoding, and basis encoding.
        
        The quantum advantage of feature maps comes from the exponential size of the Hilbert space. A classical feature vector with n dimensions can be encoded into a quantum state in a 2^n-dimensional Hilbert space. This exponential expansion allows quantum feature maps to capture complex patterns that would be difficult to represent classically.
        
        Havlíček et al. (2019) demonstrated quantum advantage using quantum feature maps for binary classification tasks. Their approach achieved higher accuracy than classical support vector machines on synthetic datasets designed to be difficult for classical algorithms. This result established quantum feature maps as a viable approach for quantum machine learning applications.
        
        Hybrid Quantum-Classical Methods:
        Hybrid quantum-classical methods combine quantum and classical computing resources to solve machine learning problems. These approaches are particularly important given the current limitations of quantum hardware. Hybrid methods typically use quantum computers for specific computational tasks while relying on classical computers for optimization, data preprocessing, and result interpretation.
        
        The variational quantum eigensolver (VQE) is a prominent example of hybrid quantum-classical methods. VQE uses quantum computers to prepare quantum states and measure observables, while classical computers optimize the parameters of the quantum circuit. This approach has been successfully applied to problems in quantum chemistry and optimization.
        
        In the context of machine learning, hybrid methods have been developed for classification, regression, and clustering tasks. These methods often achieve better performance than purely classical approaches while remaining compatible with current quantum hardware limitations.
        
        Results and Discussion:
        Our analysis of quantum machine learning research reveals several key trends and insights. First, quantum machine learning is rapidly evolving with significant progress in algorithm design and experimental validation. Second, quantum advantage has been demonstrated for specific problem classes, particularly in optimization and classification tasks. Third, hybrid quantum-classical methods show the most promise for near-term applications.
        
        The performance of quantum machine learning algorithms varies significantly depending on the problem domain and implementation approach. Quantum neural networks show promise for optimization problems but face challenges in training and generalization. Quantum feature maps offer a practical path to quantum advantage but require careful design of encoding strategies.
        
        Hardware limitations remain a significant barrier to widespread adoption of quantum machine learning. Current quantum computers have limited coherence times and high error rates, which restrict the complexity of algorithms that can be implemented. However, ongoing improvements in quantum hardware are expected to address many of these limitations.
        
        Future Directions:
        Several promising directions for future research in quantum machine learning have emerged from our analysis. First, developing quantum algorithms for unsupervised learning tasks represents an important opportunity. Current research has focused primarily on supervised learning, but unsupervised learning may offer more opportunities for quantum advantage.
        
        Second, improving the training of quantum neural networks is critical for practical applications. The barren plateau problem and other training challenges need to be addressed through better optimization strategies and circuit design. Research in quantum natural gradients and other quantum optimization techniques shows promise in this direction.
        
        Third, developing quantum machine learning algorithms for real-world applications is essential for demonstrating practical value. Current research has focused on synthetic datasets and theoretical analysis. Moving to real-world applications will require addressing issues of data encoding, error mitigation, and algorithm robustness.
        
        Conclusion:
        Quantum machine learning represents a promising new paradigm for artificial intelligence with the potential to solve problems that are intractable for classical approaches. While significant challenges remain, particularly in hardware limitations and algorithm training, the field has made remarkable progress in recent years.
        
        The development of quantum machine learning will require continued collaboration between quantum computing and machine learning researchers. Theoretical advances in algorithm design must be matched by improvements in quantum hardware and experimental validation. Hybrid quantum-classical methods offer the most practical path forward in the near term.
        
        As quantum hardware continues to improve, we expect to see broader adoption of quantum machine learning techniques across various domains. The combination of quantum computational advantages with classical machine learning frameworks has the potential to revolutionize fields ranging from drug discovery to financial modeling.
        
        References:
        1. Biamonte, J., et al. (2017). Quantum machine learning. Nature, 549(7671), 195-202.
        2. Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm. arXiv preprint arXiv:1411.4028.
        3. Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. Nature, 567(7747), 209-212.
        4. McClean, J. R., et al. (2018). Barren plateaus in quantum neural network training landscapes. Nature communications, 9(1), 1-6.
        5. Mitarai, K., et al. (2018). Quantum circuit learning. Physical Review A, 98(3), 032309.
        6. Schuld, M., & Petruccione, F. (2018). Supervised learning with quantum computers. Springer.
        7. Schuld, M., et al. (2020). Circuit-centric quantum classifiers. Physical Review A, 101(3), 032308.
        """,
        
        "ai_ethics_research": """
        Title: Ethical Considerations in Artificial Intelligence: A Framework for Responsible Development
        Authors: Dr. James Thompson, Dr. Lisa Chen, Dr. Robert Kim
        Institution: MIT, Department of Philosophy and Computer Science
        Publication Date: February 2024
        Journal: Science and Engineering Ethics
        DOI: 10.1007/s11948-024-00001-2
        
        Abstract:
        As artificial intelligence systems become increasingly sophisticated and integrated into society, addressing ethical considerations becomes paramount. This paper presents a comprehensive framework for responsible AI development that addresses issues of fairness, transparency, accountability, and societal impact. We analyze current approaches to AI ethics and propose practical guidelines for developers, policymakers, and organizations deploying AI systems.
        
        Introduction:
        The rapid advancement of artificial intelligence technology has raised profound ethical questions about its development and deployment. AI systems are now making decisions that affect millions of people in areas ranging from healthcare and criminal justice to employment and financial services. These developments require careful consideration of ethical principles and their practical implementation.
        
        Current approaches to AI ethics often focus on high-level principles without providing concrete guidance for implementation. This gap between principle and practice has led to inconsistent application of ethical standards across different AI systems and organizations. Our framework addresses this challenge by providing actionable guidelines for ethical AI development.
        
        The importance of AI ethics extends beyond individual applications to broader societal implications. As AI systems become more autonomous and capable, questions arise about responsibility, control, and the distribution of benefits and risks. These issues require interdisciplinary approaches that combine technical expertise with philosophical, legal, and social science perspectives.
        
        Ethical Principles:
        Our framework is based on five core ethical principles: fairness, transparency, accountability, privacy, and societal benefit. These principles provide a foundation for evaluating AI systems and guiding their development. Each principle is operationalized through specific criteria and assessment methods.
        
        Fairness requires that AI systems treat individuals and groups equitably, avoiding discrimination based on protected characteristics such as race, gender, age, or socioeconomic status. This principle is particularly important in high-stakes applications such as hiring, lending, and criminal justice. Fairness can be assessed through statistical measures of disparate impact and individual fairness metrics.
        
        Transparency involves making AI systems understandable to users and stakeholders. This includes explaining how decisions are made, what data is used, and what factors influence outcomes. Transparency is essential for building trust and enabling meaningful human oversight. However, the complexity of many AI systems creates challenges for achieving meaningful transparency.
        
        Accountability ensures that responsibility for AI system behavior can be assigned to appropriate parties. This includes both technical accountability for system performance and organizational accountability for deployment decisions. Accountability mechanisms should provide clear lines of responsibility and effective remedies for harms caused by AI systems.
        
        Privacy protection is essential given the sensitive nature of data used to train and operate AI systems. Privacy considerations extend beyond data protection to include the potential for AI systems to infer sensitive information from seemingly innocuous data. Effective privacy protection requires both technical safeguards and organizational policies.
        
        Societal benefit requires that AI systems contribute positively to human well-being and social progress. This principle goes beyond avoiding harm to actively promoting beneficial outcomes. Assessing societal benefit requires consideration of both direct effects and broader social implications.
        
        Implementation Framework:
        Our implementation framework provides practical guidance for applying ethical principles throughout the AI development lifecycle. The framework consists of four phases: design, development, deployment, and monitoring. Each phase includes specific activities and checkpoints for ensuring ethical compliance.
        
        The design phase focuses on identifying potential ethical issues and developing mitigation strategies. This includes conducting ethical impact assessments, engaging with stakeholders, and establishing ethical requirements. Design decisions should consider not only technical feasibility but also ethical implications and societal impact.
        
        During development, ethical considerations should be integrated into technical implementation. This includes using appropriate data sources, implementing fairness metrics, and building transparency mechanisms. Development teams should include members with expertise in relevant ethical domains and should regularly review progress against ethical requirements.
        
        The deployment phase involves careful planning for system introduction and ongoing monitoring. This includes training users, establishing oversight mechanisms, and developing response plans for ethical issues. Deployment should be gradual and monitored to identify and address unexpected ethical challenges.
        
        Ongoing monitoring is essential for identifying ethical issues that may emerge during system operation. This includes tracking system performance, user feedback, and societal impact. Monitoring should be continuous and should trigger review and revision when ethical concerns arise.
        
        Case Studies:
        We present three case studies that illustrate the application of our ethical framework in different domains. These case studies demonstrate both successful implementation and challenges encountered in practice.
        
        Case Study 1: Healthcare AI
        Our first case study examines the development of an AI system for medical diagnosis. The system was designed to assist radiologists in detecting early-stage cancer from medical images. Ethical considerations included ensuring fairness across different demographic groups, maintaining transparency about system limitations, and establishing clear accountability for diagnostic decisions.
        
        The development team implemented several ethical safeguards including bias testing on diverse datasets, clear documentation of system capabilities and limitations, and integration with existing clinical workflows to maintain human oversight. The system was deployed in a controlled clinical trial with ongoing monitoring of accuracy and fairness metrics.
        
        Results showed that the AI system improved diagnostic accuracy while maintaining fairness across demographic groups. However, challenges emerged in maintaining transparency about system reasoning and ensuring appropriate human oversight. These challenges led to revisions in the system design and deployment protocols.
        
        Case Study 2: Criminal Justice AI
        Our second case study examines the use of AI in criminal justice risk assessment. The system was designed to assess the likelihood of recidivism for individuals being considered for parole. Ethical considerations included ensuring fairness across racial and socioeconomic groups, maintaining transparency about assessment factors, and establishing accountability for decisions influenced by AI recommendations.
        
        The development team faced significant challenges in achieving fairness due to historical biases in criminal justice data. They implemented several mitigation strategies including bias correction algorithms, diverse training data, and regular fairness audits. Transparency was achieved through detailed documentation of assessment factors and their relative importance.
        
        Despite these efforts, the system showed persistent disparities in accuracy across demographic groups. This led to a decision to limit the system's use to low-risk cases and to implement additional human oversight for high-risk decisions. The case study illustrates the importance of ongoing monitoring and the need to balance competing ethical considerations.
        
        Case Study 3: Employment AI
        Our third case study examines the use of AI in hiring and promotion decisions. The system was designed to screen job applications and identify candidates most likely to succeed in specific roles. Ethical considerations included ensuring fairness across protected characteristics, maintaining transparency about decision factors, and protecting applicant privacy.
        
        The development team implemented comprehensive fairness testing and bias mitigation strategies. They also developed detailed transparency mechanisms that explained how decisions were made and what factors influenced outcomes. Privacy protection included data minimization and secure handling of sensitive information.
        
        The system was successfully deployed with high user satisfaction and improved hiring outcomes. However, ongoing monitoring revealed subtle biases that required continuous refinement of the system. The case study demonstrates the importance of long-term commitment to ethical AI development.
        
        Challenges and Limitations:
        While our framework provides a comprehensive approach to AI ethics, several challenges and limitations should be acknowledged. First, ethical principles can conflict with each other, requiring difficult trade-offs. For example, transparency may conflict with privacy protection, and fairness may conflict with accuracy in some cases.
        
        Second, the rapid pace of AI development creates challenges for ethical frameworks that may become outdated quickly. Our framework emphasizes flexibility and continuous improvement, but maintaining relevance requires ongoing attention to emerging technologies and applications.
        
        Third, implementing ethical principles requires resources and expertise that may not be available to all organizations. Smaller organizations and developing countries may face particular challenges in implementing comprehensive ethical frameworks. Addressing these challenges requires international cooperation and capacity building.
        
        Fourth, ethical considerations vary across cultures and contexts, making universal frameworks difficult to implement. Our framework emphasizes the importance of local adaptation and stakeholder engagement, but achieving consistency across different contexts remains challenging.
        
        Future Directions:
        Several promising directions for future research in AI ethics have emerged from our work. First, developing more sophisticated methods for measuring and ensuring fairness in AI systems is essential. Current fairness metrics have limitations, and new approaches are needed for complex, multi-dimensional fairness considerations.
        
        Second, improving transparency mechanisms for complex AI systems is critical for building trust and enabling meaningful human oversight. This includes developing methods for explaining AI decisions that are accessible to non-technical users and stakeholders.
        
        Third, establishing effective accountability mechanisms for AI systems requires ongoing research and development. This includes both technical approaches to tracking system behavior and organizational approaches to assigning responsibility and providing remedies.
        
        Fourth, addressing the global dimensions of AI ethics requires international cooperation and coordination. This includes developing shared standards, building capacity in developing countries, and addressing cross-border implications of AI deployment.
        
        Conclusion:
        Ethical considerations in artificial intelligence are complex and multifaceted, requiring comprehensive frameworks that address both principles and practical implementation. Our framework provides a structured approach to ethical AI development that can be adapted to different contexts and applications.
        
        The successful implementation of ethical AI requires commitment from all stakeholders including developers, organizations, policymakers, and society at large. This commitment must be ongoing, as ethical challenges evolve with technology and society. Regular review and revision of ethical frameworks is essential for maintaining their relevance and effectiveness.
        
        As AI technology continues to advance, the importance of ethical considerations will only increase. The decisions we make today about AI ethics will shape the future of technology and society. By developing and implementing comprehensive ethical frameworks, we can ensure that AI contributes positively to human well-being and social progress.
        
        The path forward requires continued research, dialogue, and collaboration across disciplines and sectors. By working together to address ethical challenges, we can build AI systems that are not only technically advanced but also ethically sound and socially beneficial.
        
        References:
        1. Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. California Law Review, 104(3), 671-732.
        2. Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608.
        3. Floridi, L., et al. (2018). AI4People—An ethical framework for a good AI society: Opportunities, risks, principles, and recommendations. Minds and Machines, 28(4), 689-707.
        4. Mittelstadt, B. D., et al. (2016). The ethics of algorithms: Mapping the debate. Big Data & Society, 3(2), 2053951716679679.
        5. Russell, S. (2019). Human compatible: Artificial intelligence and the problem of control. Penguin.
        6. Selbst, A. D., et al. (2019). Fairness and abstraction in sociotechnical systems. In Proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 59-68).
        7. Wachter, S., et al. (2017). Why a right to explanation of automated decision-making does not exist in the General Data Protection Regulation. International Data Privacy Law, 7(2), 76-99.
        """
    }
    
    return papers

def demo_academic_processing():
    """Demonstrate processing academic papers"""
    print("📚 EchoGem Academic Paper Demo")
    print("=" * 45)
    print("This demo showcases processing and analyzing academic research papers!")
    print()
    
    # Initialize processor
    print("1️⃣ Initializing EchoGem Processor...")
    try:
        processor = Processor()
        print("   ✅ Processor initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize processor: {e}")
        print("   💡 Make sure your API keys are set correctly")
        return None
    
    # Create sample papers
    print("\n2️⃣ Creating sample academic papers...")
    papers = create_academic_papers()
    for paper_name, content in papers.items():
        print(f"   📄 {paper_name}: {len(content)} characters")
    
    return processor, papers

def demo_paper_analysis(processor, papers):
    """Demonstrate academic paper analysis"""
    print("\n3️⃣ Academic Paper Analysis")
    print("=" * 35)
    
    # Process papers with academic-optimized settings
    print("   📝 Processing papers with academic-optimized settings...")
    
    academic_options = ChunkingOptions(
        max_chunk_size=600,  # Larger chunks for academic content
        overlap=100,          # More overlap for context preservation
        semantic_chunking=True
    )
    
    processed_papers = {}
    for paper_name, content in papers.items():
        print(f"\n   🔬 Processing: {paper_name}")
        start_time = time.time()
        
        try:
            response = processor.process_transcript(content, chunking_options=academic_options)
            processing_time = time.time() - start_time
            
            print(f"      ✅ Created {len(response.chunks)} chunks in {processing_time:.2f}s")
            print(f"      📊 Average chunk size: {sum(len(c.content) for c in response.chunks) // len(response.chunks)} chars")
            
            # Analyze chunk structure
            chunk_types = {}
            for chunk in response.chunks:
                chunk_type = chunk.title.split(':')[0] if ':' in chunk.title else 'General'
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            print(f"      🏗️  Chunk structure:")
            for chunk_type, count in chunk_types.items():
                print(f"         {chunk_type}: {count} chunks")
            
            processed_papers[paper_name] = response
            
        except Exception as e:
            print(f"      ❌ Processing failed: {e}")
    
    return processed_papers

def demo_academic_queries(processor, processed_papers):
    """Demonstrate academic-specific queries"""
    print("\n4️⃣ Academic-Specific Queries")
    print("=" * 40)
    
    # Define academic research questions
    research_questions = [
        # Quantum ML paper questions
        ("Quantum ML Methodology", "What methodology was used in the quantum machine learning research?"),
        ("Quantum ML Results", "What are the key results and findings in quantum machine learning?"),
        ("Quantum ML Challenges", "What challenges and limitations were identified in quantum machine learning?"),
        ("Quantum ML Future", "What future directions were suggested for quantum machine learning research?"),
        
        # AI Ethics paper questions
        ("AI Ethics Principles", "What are the core ethical principles in the AI ethics framework?"),
        ("AI Ethics Implementation", "How is the ethical framework implemented in practice?"),
        ("AI Ethics Case Studies", "What case studies were presented and what were the outcomes?"),
        ("AI Ethics Challenges", "What challenges and limitations were identified in AI ethics implementation?"),
        
        # Cross-paper questions
        ("Research Trends", "What are the common research trends across both papers?"),
        ("Methodology Comparison", "How do the research methodologies differ between the papers?"),
        ("Future Directions", "What are the overlapping future research directions?"),
        ("Impact Assessment", "What is the potential societal impact of the research described?")
    ]
    
    query_options = QueryOptions(
        show_chunks=True,
        show_prompt_answers=True,
        max_chunks=4
    )
    
    for question_name, question_text in research_questions:
        print(f"\n   🤔 {question_name}: {question_text}")
        
        start_time = time.time()
        try:
            result = processor.query(question_text, query_options=query_options)
            query_time = time.time() - start_time
            
            print(f"      ⏱️  Response time: {query_time:.2f}s")
            print(f"      📝 Answer: {result.answer[:150]}...")
            
            if hasattr(result, 'chunks_used') and result.chunks_used:
                print(f"      🔗 Chunks used: {len(result.chunks_used)}")
                # Show which papers the chunks came from
                paper_sources = set()
                for chunk in result.chunks_used:
                    if hasattr(chunk, 'metadata') and chunk.metadata:
                        source = chunk.metadata.get('source', 'Unknown')
                        paper_sources.add(source)
                if paper_sources:
                    print(f"      📚 Paper sources: {', '.join(paper_sources)}")
            
        except Exception as e:
            print(f"      ❌ Query failed: {e}")

def demo_citation_analysis(processor, processed_papers):
    """Demonstrate citation and reference analysis"""
    print("\n5️⃣ Citation and Reference Analysis")
    print("=" * 45)
    
    # Query for citations and references
    citation_queries = [
        "What are the key references cited in the quantum machine learning paper?",
        "Which authors are most frequently cited across both papers?",
        "What are the main research areas covered in the references?",
        "How recent are the references in both papers?",
        "What are the most influential papers cited in the AI ethics research?"
    ]
    
    for query in citation_queries:
        print(f"\n   📖 {query}")
        
        try:
            result = processor.query(query, query_options=QueryOptions(max_chunks=3))
            print(f"      📝 Answer: {result.answer[:120]}...")
        except Exception as e:
            print(f"      ❌ Query failed: {e}")

def demo_methodology_comparison(processor, processed_papers):
    """Demonstrate methodology comparison across papers"""
    print("\n6️⃣ Methodology Comparison")
    print("=" * 40)
    
    # Compare methodologies
    comparison_queries = [
        "How does the research methodology differ between the quantum machine learning and AI ethics papers?",
        "What are the common elements in the research approaches used in both papers?",
        "How do the case study approaches differ between the two papers?",
        "What are the different types of data analysis used in each paper?",
        "How do the validation and testing approaches compare between the papers?"
    ]
    
    for query in comparison_queries:
        print(f"\n   🔍 {query}")
        
        try:
            result = processor.query(query, query_options=QueryOptions(max_chunks=4))
            print(f"      📝 Answer: {result.answer[:120]}...")
        except Exception as e:
            print(f"      ❌ Query failed: {e}")

def demo_research_insights(processor, processed_papers):
    """Demonstrate extracting research insights"""
    print("\n7️⃣ Research Insights Extraction")
    print("=" * 40)
    
    # Extract insights
    insight_queries = [
        "What are the most significant breakthroughs described in the quantum machine learning paper?",
        "What are the key ethical challenges identified in AI development?",
        "What are the practical implications of the research findings?",
        "What are the limitations of current approaches described in both papers?",
        "What are the recommendations for future research and development?"
    ]
    
    for query in insight_queries:
        print(f"\n   💡 {query}")
        
        try:
            result = processor.query(query, query_options=QueryOptions(max_chunks=3))
            print(f"      📝 Answer: {result.answer[:120]}...")
        except Exception as e:
            print(f"      ❌ Query failed: {e}")

def main():
    """Main academic paper demo function"""
    print("🎯 EchoGem Academic Paper Processing Demonstration")
    print("=" * 70)
    print("This demo showcases how EchoGem can process and analyze academic research papers!")
    print()
    
    # Run basic demo
    result = demo_academic_processing()
    if not result:
        print("\n❌ Academic paper demo failed. Please check your setup and try again.")
        return
    
    processor, papers = result
    
    # Process papers
    processed_papers = demo_paper_analysis(processor, papers)
    
    # Run analysis demos
    demo_academic_queries(processor, processed_papers)
    demo_citation_analysis(processor, processed_papers)
    demo_methodology_comparison(processor, processed_papers)
    demo_research_insights(processor, processed_papers)
    
    # Final recommendations
    print("\n🎉 Academic Paper Demo Complete!")
    print("=" * 35)
    print("💡 Key insights for academic paper processing:")
    print("   📚 Use larger chunk sizes for academic content")
    print("   🔬 Preserve context with adequate overlap")
    print("   📖 Leverage semantic chunking for structure preservation")
    print("   🔍 Ask specific research questions for better results")
    print("   📊 Analyze chunk structure for content organization")
    
    print("\n📚 Explore other demos:")
    print("   - Basic workflow: python demos/01_basic_workflow_demo.py")
    print("   - CLI usage: python demos/02_cli_demo.py")
    print("   - Python API: python demos/03_api_demo.py")
    print("   - Performance: python demos/09_performance_benchmarking_demo.py")

if __name__ == "__main__":
    main()
