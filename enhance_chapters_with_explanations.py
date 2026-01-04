#!/usr/bin/env python3
"""
Enhance all comprehensive chapter HTML files with detailed step-by-step explanations
for all Mermaid diagrams and fix any syntax errors.
"""

import re
import os

def add_diagram_explanation(diagram_type, diagram_content):
    """Generate detailed explanation based on diagram type and content"""
    explanations = {
        'graph': 'This diagram shows the relationships and flow between different components. Follow the arrows to understand how elements connect.',
        'flowchart': 'This flowchart illustrates a step-by-step process. Start from the top and follow the arrows through each step.',
        'sequenceDiagram': 'This sequence diagram shows the interaction between different components over time. Read from top to bottom to see the order of operations.',
        'pie': 'This pie chart shows the proportional distribution of different categories.',
        'gantt': 'This Gantt chart shows a timeline of activities and their durations.'
    }
    
    base_explanation = explanations.get(diagram_type, 'This diagram visualizes the concept being discussed.')
    
    return f'''
    <div class="diagram-explanation">
        <h4>Understanding This Diagram:</h4>
        <p>{base_explanation}</p>
        <div class="step-by-step">
            <h5>Step-by-Step Breakdown:</h5>
            <ol>
    '''

def enhance_chapter_html(filepath):
    """Enhance a single chapter HTML file with detailed explanations"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all Mermaid diagrams
    mermaid_pattern = r'<div class="diagram-container"><div class="mermaid">\s*(.*?)\s*</div></div>'
    
    def enhance_diagram(match):
        diagram_code = match.group(1)
        
        # Determine diagram type
        diagram_type = 'graph'
        if 'sequenceDiagram' in diagram_code:
            diagram_type = 'sequenceDiagram'
        elif 'flowchart' in diagram_code:
            diagram_type = 'flowchart'
        elif 'pie' in diagram_code:
            diagram_type = 'pie'
        elif 'gantt' in diagram_code:
            diagram_type = 'gantt'
        
        # Generate explanation based on content
        explanation = generate_detailed_explanation(diagram_code, diagram_type)
        
        return f'''
        <div class="diagram-section">
            {explanation}
            <div class="diagram-container">
                <div class="mermaid">
{diagram_code}
                </div>
            </div>
        </div>
        '''
    
    # Replace all diagrams with enhanced versions
    enhanced_content = re.sub(mermaid_pattern, enhance_diagram, content, flags=re.DOTALL)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(enhanced_content)
    
    print(f"Enhanced {filepath}")

def generate_detailed_explanation(diagram_code, diagram_type):
    """Generate detailed step-by-step explanation for a diagram"""
    
    if diagram_type == 'sequenceDiagram':
        return generate_sequence_explanation(diagram_code)
    elif diagram_type == 'flowchart':
        return generate_flowchart_explanation(diagram_code)
    elif diagram_type == 'graph':
        return generate_graph_explanation(diagram_code)
    else:
        return '<div class="diagram-explanation"><p>This diagram visualizes important concepts.</p></div>'

def generate_sequence_explanation(code):
    """Generate explanation for sequence diagram"""
    participants = re.findall(r'participant\s+(\w+)\s+as\s+(.+?)(?:\n|$)', code)
    messages = re.findall(r'(\w+)->>(\w+):\s*(.+?)(?:\n|$)', code)
    notes = re.findall(r'Note over (.+?):\s*(.+?)(?:\n|$)', code)
    loops = re.findall(r'loop\s+(.+?)(?:\n|$)', code)
    
    explanation = '''
    <div class="diagram-explanation">
        <h4>📊 Understanding This Sequence Diagram:</h4>
        <p><strong>What is a Sequence Diagram?</strong> This diagram shows how different components interact with each other over time. Read from top to bottom to follow the sequence of operations.</p>
        
        <div class="step-by-step">
            <h5>🔍 Step-by-Step Breakdown:</h5>
            <ol>
    '''
    
    if participants:
        explanation += '<li><strong>Participants:</strong> This diagram involves the following components:<ul>'
        for p in participants:
            explanation += f'<li><strong>{p[0]}</strong>: {p[1]}</li>'
        explanation += '</ul></li>'
    
    if loops:
        explanation += f'<li><strong>Loop:</strong> The process repeats for "{loops[0]}"</li>'
    
    step_num = 2 if participants else 1
    for msg in messages[:10]:  # Limit to first 10 messages
        explanation += f'<li><strong>Step {step_num}:</strong> <strong>{msg[0]}</strong> sends "{msg[2]}" to <strong>{msg[1]}</strong></li>'
        step_num += 1
    
    if notes:
        explanation += '<li><strong>Important Notes:</strong><ul>'
        for note in notes:
            explanation += f'<li>{note[1]}</li>'
        explanation += '</ul></li>'
    
    explanation += '''
            </ol>
        </div>
        <p><strong>💡 Key Takeaway:</strong> Follow the arrows from top to bottom to understand the complete flow of operations.</p>
    </div>
    '''
    
    return explanation

def generate_flowchart_explanation(code):
    """Generate explanation for flowchart"""
    nodes = re.findall(r'(\w+)\[(.+?)\]', code)
    arrows = re.findall(r'(\w+)\s*-->\s*(\w+)', code)
    
    explanation = '''
    <div class="diagram-explanation">
        <h4>📊 Understanding This Flowchart:</h4>
        <p><strong>What is a Flowchart?</strong> This diagram shows a step-by-step process or decision flow. Start from the top and follow the arrows to understand the complete workflow.</p>
        
        <div class="step-by-step">
            <h5>🔍 Step-by-Step Breakdown:</h5>
            <ol>
    '''
    
    # Find start node (usually first node or node with "Start")
    start_nodes = [n for n in nodes if 'start' in n[1].lower() or 'begin' in n[1].lower() or 'input' in n[1].lower()]
    if start_nodes:
        explanation += f'<li><strong>Step 1 - Start:</strong> {start_nodes[0][1]}</li>'
    
    # Follow the flow
    step = 2
    for i, arrow in enumerate(arrows[:15]):  # Limit to first 15 steps
        from_node = arrow[0]
        to_node = arrow[1]
        to_label = next((n[1] for n in nodes if n[0] == to_node), to_node)
        explanation += f'<li><strong>Step {step}:</strong> From <strong>{from_node}</strong> proceed to <strong>{to_node}</strong> - {to_label}</li>'
        step += 1
    
    explanation += '''
            </ol>
        </div>
        <p><strong>💡 Key Takeaway:</strong> Follow the arrows from the start to understand the complete process flow. Decision points (diamond shapes) represent choices in the process.</p>
    </div>
    '''
    
    return explanation

def generate_graph_explanation(code):
    """Generate explanation for graph diagram"""
    nodes = re.findall(r'(\w+)\[(.+?)\]', code)
    arrows = re.findall(r'(\w+)\s*-->\s*(\w+)', code)
    
    explanation = '''
    <div class="diagram-explanation">
        <h4>📊 Understanding This Diagram:</h4>
        <p><strong>What does this show?</strong> This diagram illustrates the relationships and hierarchy between different concepts or components.</p>
        
        <div class="step-by-step">
            <h5>🔍 Step-by-Step Breakdown:</h5>
            <ol>
    '''
    
    if nodes:
        explanation += '<li><strong>Main Components:</strong><ul>'
        for node in nodes[:10]:
            explanation += f'<li><strong>{node[0]}</strong>: {node[1]}</li>'
        explanation += '</ul></li>'
    
    if arrows:
        explanation += '<li><strong>Relationships:</strong> The arrows show how components connect:<ul>'
        for arrow in arrows[:10]:
            explanation += f'<li><strong>{arrow[0]}</strong> connects to <strong>{arrow[1]}</strong></li>'
        explanation += '</ul></li>'
    
    explanation += '''
            </ol>
        </div>
        <p><strong>💡 Key Takeaway:</strong> Follow the arrows to understand how different elements relate to each other in the overall structure.</p>
    </div>
    '''
    
    return explanation

# Process all comprehensive chapters
for i in range(1, 16):
    filepath = f'comprehensive_chapter{i}.html'
    if os.path.exists(filepath):
        enhance_chapter_html(filepath)
        print(f"✓ Enhanced chapter {i}")

print("\n✅ All chapters enhanced with detailed explanations!")

