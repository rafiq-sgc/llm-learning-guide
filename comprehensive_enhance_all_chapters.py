#!/usr/bin/env python3
"""
Comprehensive script to enhance ALL comprehensive chapter HTML files with:
1. Detailed step-by-step explanations for all Mermaid diagrams
2. Fix Mermaid syntax errors
3. Add diagram-section wrappers
4. Make everything very clear and educational
"""

import re
import os

# Diagram explanation templates based on diagram type and content
def get_diagram_explanation(diagram_code, diagram_type, context=""):
    """Generate detailed explanation for a diagram"""
    
    explanations = {
        'sequenceDiagram': generate_sequence_explanation,
        'flowchart': generate_flowchart_explanation,
        'graph': generate_graph_explanation,
        'pie': generate_pie_explanation,
        'gantt': generate_gantt_explanation
    }
    
    func = explanations.get(diagram_type, lambda x, y: default_explanation(x, y))
    return func(diagram_code, context)

def generate_sequence_explanation(code, context):
    """Generate detailed explanation for sequence diagram"""
    participants = re.findall(r'participant\s+(\w+)\s+as\s+(.+?)(?:\n|$)', code)
    messages = re.findall(r'(\w+)->>(\w+):\s*(.+?)(?:\n|$)', code)
    notes = re.findall(r'Note over (.+?):\s*(.+?)(?:\n|$)', code)
    loops = re.findall(r'loop\s+(.+?)(?:\n|$)', code)
    
    explanation = '''
    <div class="diagram-explanation">
        <h4>📊 Understanding This Sequence Diagram:</h4>
        <p><strong>What is a Sequence Diagram?</strong> This diagram shows how different components interact with each other over time. Read from top to bottom to follow the sequence of operations. Each arrow represents a message or data flow between components.</p>
        
        <div class="step-by-step">
            <h5>🔍 Step-by-Step Breakdown:</h5>
            <ol>
    '''
    
    if participants:
        explanation += '<li><strong>Participants:</strong> This diagram involves the following components:<ul>'
        for p in participants:
            explanation += f'<li><strong>{p[0]}</strong>: {p[1]}</li>'
        explanation += '</ul></li>'
    
    if notes:
        for note in notes:
            explanation += f'<li><strong>Note:</strong> {note[1]}</li>'
    
    if loops:
        explanation += f'<li><strong>Loop:</strong> The following steps repeat for "{loops[0]}"</li>'
    
    step_num = len(participants) + len(notes) + len(loops) + 1
    for i, msg in enumerate(messages[:15], step_num):
        explanation += f'<li><strong>Step {i}:</strong> <strong>{msg[0]}</strong> sends "{msg[2]}" to <strong>{msg[1]}</strong></li>'
    
    explanation += '''
            </ol>
        </div>
        <p><strong>💡 Key Takeaway:</strong> Follow the arrows from top to bottom to understand the complete flow of operations. Sequence diagrams are excellent for understanding the order of operations and data flow.</p>
    </div>
    '''
    
    return explanation

def generate_flowchart_explanation(code, context):
    """Generate detailed explanation for flowchart"""
    nodes = re.findall(r'(\w+)\[(.+?)\]', code)
    decision_nodes = re.findall(r'(\w+)\{(.+?)\}', code)
    
    explanation = '''
    <div class="diagram-explanation">
        <h4>📊 Understanding This Flowchart:</h4>
        <p><strong>What is a Flowchart?</strong> This diagram shows a step-by-step process or decision flow. Start from the top and follow the arrows to understand the complete workflow. Rectangles represent processes, diamonds represent decisions.</p>
        
        <div class="step-by-step">
            <h5>🔍 Step-by-Step Breakdown:</h5>
            <ol>
    '''
    
    # Find start node
    start_nodes = [n for n in nodes if any(word in n[1].lower() for word in ['start', 'begin', 'input', 'user'])]
    if start_nodes:
        explanation += f'<li><strong>Step 1 - Start:</strong> {start_nodes[0][1]}</li>'
    
    # Process nodes
    step = 2
    for node in nodes[:20]:
        if node[0] not in [s[0] for s in start_nodes]:
            if 'decision' not in node[1].lower() and '{' not in node[1]:
                explanation += f'<li><strong>Step {step}:</strong> {node[1]}</li>'
                step += 1
    
    if decision_nodes:
        explanation += '<li><strong>Decision Points:</strong><ul>'
        for dec in decision_nodes:
            explanation += f'<li>{dec[1]} - Choose Yes or No path</li>'
        explanation += '</ul></li>'
    
    explanation += '''
            </ol>
        </div>
        <p><strong>💡 Key Takeaway:</strong> Follow the arrows from the start to understand the complete process flow. Decision points (diamond shapes) represent choices in the process - follow the appropriate path based on the condition.</p>
    </div>
    '''
    
    return explanation

def generate_graph_explanation(code, context):
    """Generate detailed explanation for graph diagram"""
    nodes = re.findall(r'(\w+)\[(.+?)\]', code)
    arrows = re.findall(r'(\w+)\s*-->\s*(\w+)', code)
    
    explanation = '''
    <div class="diagram-explanation">
        <h4>📊 Understanding This Diagram:</h4>
        <p><strong>What does this show?</strong> This diagram illustrates the relationships and hierarchy between different concepts or components. Follow the arrows to understand how elements connect and relate to each other.</p>
        
        <div class="step-by-step">
            <h5>🔍 Step-by-Step Breakdown:</h5>
            <ol>
    '''
    
    if nodes:
        explanation += '<li><strong>Main Components:</strong><ul>'
        for node in nodes[:15]:
            explanation += f'<li><strong>{node[0]}</strong>: {node[1]}</li>'
        explanation += '</ul></li>'
    
    if arrows:
        explanation += '<li><strong>Relationships:</strong> The arrows show how components connect:<ul>'
        for arrow in arrows[:15]:
            from_label = next((n[1] for n in nodes if n[0] == arrow[0]), arrow[0])
            to_label = next((n[1] for n in nodes if n[0] == arrow[1]), arrow[1])
            explanation += f'<li><strong>{from_label}</strong> connects to <strong>{to_label}</strong></li>'
        explanation += '</ul></li>'
    
    explanation += '''
            </ol>
        </div>
        <p><strong>💡 Key Takeaway:</strong> Follow the arrows to understand how different elements relate to each other in the overall structure. The hierarchy and connections reveal the organization of concepts.</p>
    </div>
    '''
    
    return explanation

def generate_pie_explanation(code, context):
    """Generate explanation for pie chart"""
    title_match = re.search(r'title\s+(.+?)(?:\n|$)', code)
    segments = re.findall(r'"(.+?)"\s*:\s*(\d+)', code)
    
    explanation = '''
    <div class="diagram-explanation">
        <h4>📊 Understanding This Pie Chart:</h4>
        <p><strong>What does this show?</strong> This pie chart shows the proportional distribution of different categories. The size of each slice represents its proportion of the total.</p>
        
        <div class="step-by-step">
            <h5>🔍 Step-by-Step Breakdown:</h5>
            <ol>
    '''
    
    if title_match:
        explanation += f'<li><strong>Chart Title:</strong> {title_match.group(1)}</li>'
    
    if segments:
        total = sum(int(s[1]) for s in segments)
        explanation += '<li><strong>Distribution:</strong><ul>'
        for seg in segments:
            percentage = (int(seg[1]) / total * 100) if total > 0 else 0
            explanation += f'<li><strong>{seg[0]}</strong>: {seg[1]}% ({percentage:.1f}% of total)</li>'
        explanation += '</ul></li>'
    
    explanation += '''
            </ol>
        </div>
        <p><strong>💡 Key Takeaway:</strong> Larger slices represent larger proportions. This visualization helps quickly understand the relative importance or distribution of different categories.</p>
    </div>
    '''
    
    return explanation

def generate_gantt_explanation(code, context):
    """Generate explanation for Gantt chart"""
    sections = re.findall(r'section\s+(.+?)(?:\n|$)', code)
    tasks = re.findall(r'(.+?)\s*:(.+?)(?:\n|$)', code)
    
    explanation = '''
    <div class="diagram-explanation">
        <h4>📊 Understanding This Gantt Chart:</h4>
        <p><strong>What is a Gantt Chart?</strong> This chart shows a timeline of activities and their durations. Each bar represents a task, and its length shows how long it takes.</p>
        
        <div class="step-by-step">
            <h5>🔍 Step-by-Step Breakdown:</h5>
            <ol>
    '''
    
    if sections:
        explanation += '<li><strong>Sections:</strong><ul>'
        for sec in sections:
            explanation += f'<li>{sec}</li>'
        explanation += '</ul></li>'
    
    if tasks:
        explanation += '<li><strong>Tasks and Durations:</strong><ul>'
        for task in tasks[:10]:
            explanation += f'<li>{task[0]}: {task[1]}</li>'
        explanation += '</ul></li>'
    
    explanation += '''
            </ol>
        </div>
        <p><strong>💡 Key Takeaway:</strong> Read from left to right to see the timeline. Tasks that overlap happen in parallel. The total length shows the complete duration.</p>
    </div>
    '''
    
    return explanation

def default_explanation(code, context):
    """Default explanation for unknown diagram types"""
    return '''
    <div class="diagram-explanation">
        <h4>📊 Understanding This Diagram:</h4>
        <p>This diagram visualizes important concepts related to the topic. Study it carefully to understand the relationships and flow shown.</p>
        <p><strong>💡 Key Takeaway:</strong> Diagrams help visualize complex concepts. Take time to understand each element and how they connect.</p>
    </div>
    '''

def fix_mermaid_syntax(code):
    """Fix common Mermaid syntax errors"""
    # Fix Note over with commas
    code = re.sub(r'Note over ([^,]+),([^:]+):', r'Note over \1: Note also applies to \2\n    Note over \2:', code)
    
    # Fix arrow symbols in node labels (already handled in markdown, but double-check)
    code = code.replace('→', 'to')
    
    # Fix parentheses in sequence diagram messages
    code = re.sub(r'->>(\w+):\s*([^(]+)\(([^)]+)\)', r'->>\1: \2\3', code)
    
    return code

def enhance_chapter_file(filepath):
    """Enhance a single chapter HTML file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to find Mermaid diagrams
    pattern = r'<p><div class="diagram-container"><div class="mermaid">\s*(.*?)\s*</div></div></p>'
    
    def replace_diagram(match):
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
        
        # Fix syntax errors
        fixed_code = fix_mermaid_syntax(diagram_code)
        
        # Generate explanation
        explanation = get_diagram_explanation(fixed_code, diagram_type)
        
        # Return enhanced diagram
        return f'''
        <div class="diagram-section">
            {explanation}
            <div class="diagram-container">
                <div class="mermaid">
{fixed_code}
                </div>
            </div>
        </div>
        '''
    
    # Replace all diagrams
    enhanced_content = re.sub(pattern, replace_diagram, content, flags=re.DOTALL)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(enhanced_content)
    
    print(f"✓ Enhanced {filepath}")

# Process all comprehensive chapters
print("Enhancing all comprehensive chapters with detailed explanations...\n")

for i in range(1, 16):
    filepath = f'comprehensive_chapter{i}.html'
    if os.path.exists(filepath):
        try:
            enhance_chapter_file(filepath)
        except Exception as e:
            print(f"✗ Error enhancing {filepath}: {e}")

print("\n✅ All chapters enhanced with detailed explanations!")
print("📝 Note: Some diagrams may need manual review for context-specific explanations.")

