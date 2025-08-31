#!/usr/bin/env python3
"""
EchoGem Meeting Transcript Demo

This demo showcases processing and analyzing business meeting transcripts:
- Meeting structure analysis
- Action item extraction
- Decision tracking
- Participant contribution analysis
- Timeline and agenda analysis
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

def create_meeting_transcripts():
    """Create sample meeting transcripts for demonstration"""
    meetings = {
        "q4_planning": """
        Q4 2024 Planning Meeting
        Date: December 15, 2024
        Time: 10:00 AM - 11:30 AM
        Attendees: Sarah Johnson (CEO), Mike Chen (CTO), Lisa Rodriguez (CFO), Tom Wilson (VP Sales), Emma Davis (VP Marketing)
        Facilitator: Sarah Johnson
        
        Sarah: Good morning everyone. Welcome to our Q4 planning session. We have a lot to cover today, so let's get started. First, let me give you a quick overview of our Q3 performance and then we'll dive into Q4 planning.
        
        Q3 Performance Review
        Sarah: Q3 was a strong quarter for us. We exceeded our revenue target by 8%, achieving $15.2M against our $14.1M goal. Our customer acquisition cost decreased by 12%, and our customer lifetime value increased by 18%. However, we did miss our profit margin target by 2 percentage points due to increased infrastructure costs.
        
        Mike: On the technical side, we successfully launched three major features: the new analytics dashboard, mobile app redesign, and API v3.0. User engagement increased by 25% after the dashboard launch, and mobile app crash rates decreased by 40%. However, we're still behind on the machine learning integration project.
        
        Lisa: From a financial perspective, our cash flow remains strong with $8.5M in reserves. We've managed to reduce operational expenses by 5% through process optimization, but we need to address the infrastructure cost increases that Mike mentioned.
        
        Q4 Goals and Objectives
        Sarah: For Q4, we have three primary objectives: achieve $18M in revenue, launch the enterprise security features, and expand to two new markets. Let's break these down.
        
        Revenue Target Discussion
        Tom: The $18M target is ambitious but achievable. We have a strong pipeline with $12M in qualified opportunities. The key will be closing the enterprise deals we've been working on. I'm confident we can hit this target if we focus on our top 20 prospects.
        
        Emma: Marketing is aligned with this goal. We're launching a new campaign targeting enterprise decision-makers, and we've increased our digital advertising budget by 30%. We're also planning a major product launch event in early December.
        
        Lisa: I support the revenue target, but we need to ensure we're not sacrificing profitability. I recommend we maintain our current pricing structure and focus on value-based selling rather than discounting.
        
        Enterprise Security Features
        Mike: The enterprise security features are 70% complete. We're implementing SSO, advanced role-based access control, and audit logging. The main challenge is the third-party security audit, which is scheduled for the last week of November.
        
        Sarah: What's the timeline for completion?
        Mike: We can complete development by November 20th, but the security audit will take 2-3 weeks. Realistically, we're looking at mid-December for full deployment.
        
        Market Expansion
        Sarah: For market expansion, we're targeting Southeast Asia and Latin America. Emma, what's the status on these markets?
        
        Emma: Southeast Asia is ready to go. We've completed market research, identified key partners, and have a local team in place. We can launch there by December 1st. Latin America is more complex due to regulatory requirements, but we're targeting January 15th.
        
        Tom: I'm concerned about the Latin America timeline. The regulatory approval process can take longer than expected. I recommend we focus on Southeast Asia for Q4 and push Latin America to Q1.
        
        Resource Allocation
        Sarah: Good point, Tom. Let's adjust our plan. Southeast Asia in Q4, Latin America in Q1. Now let's talk about resource allocation. What do we need to achieve these goals?
        
        Mike: For the security features, I need two additional developers for the next six weeks. We also need to increase our QA testing capacity by 50% to handle the security audit requirements.
        
        Emma: Marketing needs additional budget for the enterprise campaign and the product launch event. I'm requesting an additional $500K for Q4.
        
        Lisa: I can approve the additional marketing budget, but we need to find cost savings elsewhere to maintain our profitability targets. I suggest we delay some non-critical infrastructure upgrades.
        
        Risk Assessment
        Sarah: Let's identify the key risks to our Q4 plan. What are the biggest concerns?
        
        Mike: The biggest technical risk is the security audit. If we fail, it could delay the enterprise features launch by several weeks. We also need to ensure our infrastructure can handle the increased load from new markets.
        
        Tom: From a sales perspective, the main risk is the enterprise deals not closing on time. These are complex sales cycles, and any delays could impact our revenue target.
        
        Lisa: Financial risks include increased marketing spend without guaranteed returns, and potential infrastructure cost overruns. We need to monitor these closely.
        
        Action Items and Next Steps
        Sarah: Let's summarize our action items and next steps.
        
        Mike: I'll complete the security features by November 20th and prepare for the security audit. I'll also hire the two additional developers this week.
        
        Emma: I'll launch the Southeast Asia market on December 1st and prepare the enterprise marketing campaign. I'll also plan the product launch event.
        
        Tom: I'll focus on closing the top 20 enterprise deals and prepare the Latin America market launch plan for Q1.
        
        Lisa: I'll monitor our financial metrics weekly and prepare contingency plans for cost overruns.
        
        Sarah: I'll coordinate with the board on our Q4 plan and ensure we have their support for the additional investments.
        
        Timeline and Milestones
        Sarah: Let's establish our key milestones for Q4.
        
        Week 1-2: Complete security features development
        Week 3-4: Security audit and testing
        Week 5: Southeast Asia market launch
        Week 6-8: Enterprise features deployment
        Week 9-12: Focus on revenue generation and deal closure
        
        Next Meeting
        Sarah: We'll have a weekly check-in every Monday at 9 AM to track progress. Our next full planning session will be on January 10th for Q1 planning. Any questions or concerns before we wrap up?
        
        Tom: Just to confirm, we're targeting $18M in revenue, launching Southeast Asia in Q4, and pushing Latin America to Q1?
        
        Sarah: Correct. Southeast Asia in Q4, Latin America in Q1. The $18M target stands, but we'll monitor progress weekly and adjust if needed.
        
        Sarah: Great. Thank you everyone for your input. Let's make Q4 our best quarter yet. Meeting adjourned.
        """,
        
        "product_strategy": """
        Product Strategy and Roadmap Meeting
        Date: December 10, 2024
        Time: 2:00 PM - 4:00 PM
        Attendees: Mike Chen (CTO), Lisa Rodriguez (CFO), Emma Davis (VP Marketing), David Kim (Product Manager), Rachel Green (UX Designer)
        Facilitator: Mike Chen
        
        Mike: Welcome everyone to our product strategy and roadmap meeting. Today we're going to review our current product performance, discuss the roadmap for the next 12 months, and align on our strategic priorities.
        
        Current Product Performance
        David: Let me start with our current product performance. Our flagship product, DataFlow Pro, has seen 34% growth in active users this quarter. The new analytics dashboard we launched in October has been particularly successful, with 78% adoption rate among enterprise customers.
        
        Rachel: From a UX perspective, the dashboard redesign has been well-received. Our user satisfaction score increased from 4.2 to 4.7 out of 5. The most requested features are better mobile experience and more customization options.
        
        Emma: Marketing has seen strong demand for the analytics features. We've received 45% more demo requests since the dashboard launch, and our conversion rate from demo to trial increased by 20%.
        
        Lisa: The financial impact has been positive. Revenue per user increased by 15%, and our customer churn rate decreased by 8%. However, our customer acquisition cost increased by 12% due to higher marketing spend.
        
        Product Roadmap Discussion
        Mike: Now let's discuss our 12-month roadmap. We have several major initiatives planned. Let me outline them:
        
        Q1 2025: Machine Learning Integration
        Q2 2025: Advanced Collaboration Features
        Q3 2025: Mobile App Redesign
        Q4 2025: Enterprise Security Suite
        
        Let's start with Q1. David, what's the status on the ML integration?
        
        David: The ML integration project is currently 40% complete. We're implementing recommendation engines, predictive analytics, and automated insights. The main challenge is integrating with our existing data infrastructure.
        
        Rachel: From a UX perspective, we need to ensure the ML features are intuitive and don't overwhelm users. I'm proposing a gradual rollout with progressive disclosure of advanced features.
        
        Mike: What's the timeline for Q1 delivery?
        
        David: We can deliver the core ML features by March 31st, but the full integration will take until mid-April. I recommend we launch the core features in Q1 and complete the integration in Q2.
        
        Q2 2025: Collaboration Features
        Emma: The collaboration features are critical for our enterprise customers. We're seeing strong demand for real-time collaboration, shared workspaces, and team management tools.
        
        David: We've completed the design phase for collaboration features. The main components are real-time editing, comment systems, and team workspaces. Development can start in January.
        
        Rachel: The UX design focuses on seamless collaboration without disrupting individual workflows. We're using a sidebar approach for collaboration tools to keep the main interface clean.
        
        Q3 2025: Mobile App Redesign
        Rachel: The mobile app redesign is our biggest UX project for 2025. We're completely rebuilding the mobile experience to match our web platform's capabilities.
        
        David: The mobile redesign will include all the features from our web platform, optimized for mobile use. We're also adding mobile-specific features like offline support and push notifications.
        
        Mike: What's the development timeline for mobile?
        
        David: Mobile development will take 6-8 weeks. We can start in July and deliver by September 30th. We'll need to hire two additional mobile developers.
        
        Q4 2025: Enterprise Security Suite
        Mike: The enterprise security suite is our most complex project. It includes advanced authentication, data encryption, compliance tools, and audit capabilities.
        
        Lisa: This is also our most expensive project. The estimated cost is $2.5M, which represents 15% of our annual R&D budget.
        
        David: The security suite will take 4-5 months to develop. We need to start development in August to meet the Q4 deadline.
        
        Resource Requirements
        Mike: Let's discuss the resource requirements for these projects. What do we need?
        
        David: For Q1 ML integration, we need one additional ML engineer and one data engineer. For Q2 collaboration, we need two full-stack developers. For Q3 mobile, we need two mobile developers. For Q4 security, we need three security-focused developers.
        
        Rachel: UX design will need additional resources too. I recommend hiring one more UX designer and one UI developer to support these projects.
        
        Lisa: What's the total cost for these resources?
        
        David: The additional headcount will cost approximately $1.8M annually. Combined with the security suite development, our total R&D investment for 2025 will be $4.3M.
        
        Lisa: That's a 25% increase over our current R&D budget. We need to ensure this investment generates sufficient returns.
        
        Strategic Priorities
        Mike: Given our resource constraints, we need to prioritize these projects. What are our strategic priorities?
        
        Emma: From a market perspective, the ML integration is critical. Our competitors are already offering ML features, and we're losing deals because we don't have them.
        
        David: I agree. ML integration should be our top priority. It will enable all our other features and give us a competitive advantage.
        
        Rachel: The mobile redesign is also critical. Mobile usage is growing rapidly, and our current mobile experience is limiting our growth.
        
        Mike: Based on this discussion, I recommend we prioritize as follows:
        1. ML Integration (Q1)
        2. Mobile Redesign (Q3)
        3. Collaboration Features (Q2)
        4. Security Suite (Q4)
        
        Lisa: I support this prioritization, but we need to ensure we have the budget. I recommend we phase the security suite development to spread the cost over 2025 and 2026.
        
        Risk Assessment
        Mike: Let's identify the key risks to our roadmap.
        
        David: The biggest technical risk is the ML integration. We're working with new technologies and there's a learning curve. We also need to ensure our data infrastructure can handle the increased load.
        
        Rachel: UX risk is high for the mobile redesign. We're making significant changes to the user experience, which could impact user adoption.
        
        Lisa: Financial risk is the increased R&D spending without guaranteed returns. We need to ensure these investments generate sufficient revenue growth.
        
        Emma: Market risk is that our competitors launch similar features before we do, reducing our competitive advantage.
        
        Success Metrics
        Mike: Let's define success metrics for each project.
        
        ML Integration: 60% user adoption within 3 months, 25% increase in user engagement
        Collaboration Features: 40% user adoption, 20% increase in team collaboration
        Mobile Redesign: 50% increase in mobile usage, 30% improvement in mobile satisfaction scores
        Security Suite: 80% enterprise customer adoption, 15% increase in enterprise revenue
        
        Next Steps
        Mike: Let's summarize our next steps.
        
        David: I'll complete the ML integration design by January 15th and start development on February 1st.
        
        Rachel: I'll complete the mobile redesign mockups by January 31st and start working with the development team in February.
        
        Emma: I'll prepare marketing plans for each feature launch and ensure we have customer feedback throughout development.
        
        Lisa: I'll prepare the budget proposal for the additional resources and present it to the board in January.
        
        Mike: I'll coordinate the hiring process and ensure we have the right team in place for each project.
        
        Next Meeting
        Mike: We'll have a bi-weekly check-in on these projects, starting January 15th. Our next full roadmap review will be in March. Any questions before we wrap up?
        
        Rachel: Just to confirm, we're prioritizing ML integration and mobile redesign over collaboration and security?
        
        Mike: Correct. ML integration and mobile redesign are our top priorities for 2025. Collaboration and security will follow based on available resources.
        
        Mike: Great. Thank you everyone for your input. Let's make 2025 our best year yet for product development. Meeting adjourned.
        """
    }
    
    return meetings

def demo_meeting_processing():
    """Demonstrate processing meeting transcripts"""
    print("📋 EchoGem Meeting Transcript Demo")
    print("=" * 45)
    print("This demo showcases processing and analyzing business meeting transcripts!")
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
    
    # Create sample meetings
    print("\n2️⃣ Creating sample meeting transcripts...")
    meetings = create_meeting_transcripts()
    for meeting_name, content in meetings.items():
        print(f"   📝 {meeting_name}: {len(content)} characters")
    
    return processor, meetings

def demo_meeting_analysis(processor, meetings):
    """Demonstrate meeting transcript analysis"""
    print("\n3️⃣ Meeting Transcript Analysis")
    print("=" * 40)
    
    # Process meetings with meeting-optimized settings
    print("   📝 Processing meetings with meeting-optimized settings...")
    
    meeting_options = ChunkingOptions(
        max_chunk_size=500,  # Medium chunks for meeting content
        overlap=75,           # Adequate overlap for context
        semantic_chunking=True
    )
    
    processed_meetings = {}
    for meeting_name, content in meetings.items():
        print(f"\n   📋 Processing: {meeting_name}")
        start_time = time.time()
        
        try:
            response = processor.process_transcript(content, chunking_options=meeting_options)
            processing_time = time.time() - start_time
            
            print(f"      ✅ Created {len(response.chunks)} chunks in {processing_time:.2f}s")
            print(f"      📊 Average chunk size: {sum(len(c.content) for c in response.chunks) // len(response.chunks)} chars")
            
            # Analyze chunk structure
            chunk_types = {}
            for chunk in response.chunks:
                # Identify chunk types based on content
                if any(keyword in chunk.content.lower() for keyword in ['action', 'next step', 'deadline']):
                    chunk_types['Action Items'] = chunk_types.get('Action Items', 0) + 1
                elif any(keyword in chunk.content.lower() for keyword in ['decision', 'agree', 'approve']):
                    chunk_types['Decisions'] = chunk_types.get('Decisions', 0) + 1
                elif any(keyword in chunk.content.lower() for keyword in ['risk', 'concern', 'challenge']):
                    chunk_types['Risks'] = chunk_types.get('Risks', 0) + 1
                elif any(keyword in chunk.content.lower() for keyword in ['timeline', 'schedule', 'milestone']):
                    chunk_types['Timeline'] = chunk_types.get('Timeline', 0) + 1
                else:
                    chunk_types['Discussion'] = chunk_types.get('Discussion', 0) + 1
            
            print(f"      🏗️  Chunk categorization:")
            for chunk_type, count in chunk_types.items():
                print(f"         {chunk_type}: {count} chunks")
            
            processed_meetings[meeting_name] = response
            
        except Exception as e:
            print(f"      ❌ Processing failed: {e}")
    
    return processed_meetings

def demo_action_item_extraction(processor, processed_meetings):
    """Demonstrate action item extraction from meetings"""
    print("\n4️⃣ Action Item Extraction")
    print("=" * 35)
    
    # Extract action items from each meeting
    for meeting_name, response in processed_meetings.items():
        print(f"\n   📋 {meeting_name} - Action Items:")
        
        action_item_queries = [
            "What are the specific action items mentioned in this meeting?",
            "Who is responsible for what actions?",
            "What are the deadlines and timelines mentioned?",
            "What are the next steps identified?",
            "What resources or support is needed for these actions?"
        ]
        
        for query in action_item_queries:
            print(f"\n      🤔 {query}")
            
            try:
                result = processor.query(query, QueryOptions(max_chunks=3))
                print(f"         📝 Answer: {result.answer[:120]}...")
                
                if hasattr(result, 'chunks_used') and result.chunks_used:
                    print(f"         🔗 Chunks used: {len(result.chunks_used)}")
                
            except Exception as e:
                print(f"         ❌ Query failed: {e}")

def demo_decision_tracking(processor, processed_meetings):
    """Demonstrate decision tracking from meetings"""
    print("\n5️⃣ Decision Tracking")
    print("=" * 30)
    
    # Track decisions from each meeting
    for meeting_name, response in processed_meetings.items():
        print(f"\n   📋 {meeting_name} - Decisions:")
        
        decision_queries = [
            "What decisions were made in this meeting?",
            "What was agreed upon or approved?",
            "What changes were decided?",
            "What priorities were established?",
            "What budget or resource decisions were made?"
        ]
        
        for query in decision_queries:
            print(f"\n      🤔 {query}")
            
            try:
                result = processor.query(query, QueryOptions(max_chunks=3))
                print(f"         📝 Answer: {result.answer[:120]}...")
                
            except Exception as e:
                print(f"         ❌ Query failed: {e}")

def demo_participant_analysis(processor, processed_meetings):
    """Demonstrate participant contribution analysis"""
    print("\n6️⃣ Participant Contribution Analysis")
    print("=" * 45)
    
    # Analyze participant contributions
    for meeting_name, response in processed_meetings.items():
        print(f"\n   📋 {meeting_name} - Participant Analysis:")
        
        participant_queries = [
            "Who were the main participants in this meeting?",
            "What were the key contributions from each participant?",
            "Who made the most decisions or had the most influence?",
            "What concerns or objections were raised by participants?",
            "How did participants collaborate or work together?"
        ]
        
        for query in participant_queries:
            print(f"\n      🤔 {query}")
            
            try:
                result = processor.query(query, QueryOptions(max_chunks=3))
                print(f"         📝 Answer: {result.answer[:120]}...")
                
            except Exception as e:
                print(f"         ❌ Query failed: {e}")

def demo_timeline_analysis(processor, processed_meetings):
    """Demonstrate timeline and agenda analysis"""
    print("\n7️⃣ Timeline and Agenda Analysis")
    print("=" * 40)
    
    # Analyze timelines and agendas
    for meeting_name, response in processed_meetings.items():
        print(f"\n   📋 {meeting_name} - Timeline Analysis:")
        
        timeline_queries = [
            "What was the agenda for this meeting?",
            "What were the key milestones or deadlines mentioned?",
            "What is the timeline for implementing decisions?",
            "What are the next meeting dates or check-ins?",
            "How was the meeting structured and organized?"
        ]
        
        for query in timeline_queries:
            print(f"\n      🤔 {query}")
            
            try:
                result = processor.query(query, QueryOptions(max_chunks=3))
                print(f"         📝 Answer: {result.answer[:120]}...")
                
            except Exception as e:
                print(f"         ❌ Query failed: {e}")

def demo_cross_meeting_insights(processor, processed_meetings):
    """Demonstrate insights across multiple meetings"""
    print("\n8️⃣ Cross-Meeting Insights")
    print("=" * 35)
    
    # Analyze patterns across meetings
    print("   🔍 Analyzing patterns across meetings...")
    
    cross_meeting_queries = [
        "What are the common themes across these meetings?",
        "How do the priorities align between meetings?",
        "What are the recurring challenges or concerns?",
        "How do the timelines and deadlines relate to each other?",
        "What are the overall strategic objectives from these meetings?"
    ]
    
    for query in cross_meeting_queries:
        print(f"\n   🤔 {query}")
        
        try:
            result = processor.query(query, QueryOptions(max_chunks=5))
            print(f"      📝 Answer: {result.answer[:150]}...")
            
        except Exception as e:
            print(f"      ❌ Query failed: {e}")

def demo_meeting_summary_generation(processor, processed_meetings):
    """Demonstrate meeting summary generation"""
    print("\n9️⃣ Meeting Summary Generation")
    print("=" * 40)
    
    # Generate comprehensive summaries
    for meeting_name, response in processed_meetings.items():
        print(f"\n   📋 {meeting_name} - Comprehensive Summary:")
        
        summary_query = f"Provide a comprehensive summary of the {meeting_name} meeting, including key decisions, action items, timelines, and next steps."
        
        try:
            result = processor.query(summary_query, QueryOptions(max_chunks=5))
            print(f"      📝 Summary: {result.answer[:200]}...")
            
        except Exception as e:
            print(f"      ❌ Summary generation failed: {e}")

def main():
    """Main meeting transcript demo function"""
    print("🎯 EchoGem Meeting Transcript Processing Demonstration")
    print("=" * 75)
    print("This demo showcases how EchoGem can process and analyze business meeting transcripts!")
    print()
    
    # Run basic demo
    result = demo_meeting_processing()
    if not result:
        print("\n❌ Meeting transcript demo failed. Please check your setup and try again.")
        return
    
    processor, meetings = result
    
    # Process meetings
    processed_meetings = demo_meeting_analysis(processor, meetings)
    
    # Run analysis demos
    demo_action_item_extraction(processor, processed_meetings)
    demo_decision_tracking(processor, processed_meetings)
    demo_participant_analysis(processor, processed_meetings)
    demo_timeline_analysis(processor, processed_meetings)
    demo_cross_meeting_insights(processor, processed_meetings)
    demo_meeting_summary_generation(processor, processed_meetings)
    
    # Final recommendations
    print("\n🎉 Meeting Transcript Demo Complete!")
    print("=" * 40)
    print("💡 Key insights for meeting transcript processing:")
    print("   📋 Use medium chunk sizes for meeting content")
    print("   🔍 Focus on action items and decisions")
    print("   👥 Analyze participant contributions")
    print("   ⏰ Track timelines and deadlines")
    print("   📊 Identify patterns across meetings")
    
    print("\n📚 Explore other demos:")
    print("   - Basic workflow: python demos/01_basic_workflow_demo.py")
    print("   - CLI usage: python demos/02_cli_demo.py")
    print("   - Python API: python demos/03_api_demo.py")
    print("   - Academic papers: python demos/04_academic_paper_demo.py")
    print("   - Performance: python demos/09_performance_benchmarking_demo.py")

if __name__ == "__main__":
    main()
