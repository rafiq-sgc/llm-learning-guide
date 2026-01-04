#!/usr/bin/env python3
"""
Generate comprehensive HTML chapters from the detailed guide
"""

import re
import os

# Read the comprehensive guide
with open('COMPREHENSIVE_LLM_GUIDE.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Chapter definitions
chapters = [
    {
        'num': 1,
        'title': 'Introduction to Artificial Intelligence',
        'icon': '🤖',
        'start': '## 1. Introduction to Artificial Intelligence',
        'end': '## 2. Machine Learning Fundamentals'
    },
    {
        'num': 2,
        'title': 'Machine Learning Fundamentals',
        'icon': '📊',
        'start': '## 2. Machine Learning Fundamentals',
        'end': '## 3. Deep Learning Basics'
    },
    {
        'num': 3,
        'title': 'Deep Learning Basics',
        'icon': '🧠',
        'start': '## 3. Deep Learning Basics',
        'end': '## 4. Neural Networks Deep Dive'
    },
    {
        'num': 4,
        'title': 'Neural Networks Deep Dive',
        'icon': '🔗',
        'start': '## 4. Neural Networks Deep Dive',
        'end': '## 5. Natural Language Processing Evolution'
    },
    {
        'num': 5,
        'title': 'Natural Language Processing Evolution',
        'icon': '💬',
        'start': '## 5. Natural Language Processing Evolution',
        'end': '## 6. The Transformer Revolution'
    },
    {
        'num': 6,
        'title': 'The Transformer Revolution',
        'icon': '⚡',
        'start': '## 6. The Transformer Revolution',
        'end': '## 7. How LLMs Are Trained'
    },
    {
        'num': 7,
        'title': 'How LLMs Are Trained - Complete Process',
        'icon': '🎓',
        'start': '## 7. How LLMs Are Trained - Complete Process',
        'end': '## 8. LLM Architecture'
    },
    {
        'num': 8,
        'title': 'LLM Architecture - Detailed Breakdown',
        'icon': '🏗️',
        'start': '## 8. LLM Architecture - Detailed Breakdown',
        'end': '## 9. How LLMs Process User Queries'
    },
    {
        'num': 9,
        'title': 'How LLMs Process User Queries - Step by Step',
        'icon': '🔄',
        'start': '## 9. How LLMs Process User Queries - Step by Step',
        'end': '## 10. Attention Mechanism'
    },
    {
        'num': 10,
        'title': 'Attention Mechanism - Deep Understanding',
        'icon': '👁️',
        'start': '## 10. Attention Mechanism - Deep Understanding',
        'end': '## 11. Training Data and Preprocessing'
    },
    {
        'num': 11,
        'title': 'Training Data and Preprocessing',
        'icon': '📚',
        'start': '## 11. Training Data and Preprocessing',
        'end': '## 12. Fine-tuning and Specialization'
    },
    {
        'num': 12,
        'title': 'Fine-tuning and Specialization',
        'icon': '🎯',
        'start': '## 12. Fine-tuning and Specialization',
        'end': '## 13. LLM Inference'
    },
    {
        'num': 13,
        'title': 'LLM Inference - Complete Flow',
        'icon': '⚙️',
        'start': '## 13. LLM Inference - Complete Flow',
        'end': '## 14. Modern LLM Evolution'
    },
    {
        'num': 14,
        'title': 'Modern LLM Evolution',
        'icon': '📈',
        'start': '## 14. Modern LLM Evolution',
        'end': '## 15. LLM Applications and Future'
    },
    {
        'num': 15,
        'title': 'LLM Applications and Future',
        'icon': '🚀',
        'start': '## 15. LLM Applications and Future',
        'end': '**End of Comprehensive Guide**'
    }
]

def extract_chapter_content(content, start_marker, end_marker):
    """Extract content between two markers"""
    start_idx = content.find(start_marker)
    if start_idx == -1:
        return None
    
    end_idx = content.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        chapter_content = content[start_idx:]
    else:
        chapter_content = content[start_idx:end_idx]
    
    return chapter_content.strip()

def markdown_to_html(text):
    """Convert markdown to HTML with proper handling"""
    html = text
    
    # Preserve mermaid diagrams first
    mermaid_blocks = []
    def save_mermaid(match):
        idx = len(mermaid_blocks)
        mermaid_blocks.append(match.group(0))
        return f"MERMAID_BLOCK_{idx}"
    
    html = re.sub(r'```mermaid\n(.*?)\n```', save_mermaid, html, flags=re.DOTALL)
    
    # Convert code blocks (non-mermaid)
    def code_block(match):
        lang = match.group(1) or ''
        code = match.group(2)
        code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return f'<pre><code>{code}</code></pre>'
    
    html = re.sub(r'```(\w+)?\n(.*?)```', code_block, html, flags=re.DOTALL)
    
    # Convert headers
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # Convert bold
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    
    # Convert inline code
    html = re.sub(r'`([^`\n]+)`', r'<code class="inline-code">\1</code>', html)
    
    # Convert lists
    lines = html.split('\n')
    result = []
    in_ul = False
    in_ol = False
    
    for line in lines:
        stripped = line.strip()
        
        if re.match(r'^- ', stripped):
            if not in_ul:
                if in_ol:
                    result.append('</ol>')
                    in_ol = False
                result.append('<ul>')
                in_ul = True
            item = re.sub(r'^- ', '', stripped)
            item = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item)
            item = re.sub(r'`([^`]+)`', r'<code class="inline-code">\1</code>', item)
            result.append(f'<li>{item}</li>')
        elif re.match(r'^\d+\. ', stripped):
            if not in_ol:
                if in_ul:
                    result.append('</ul>')
                    in_ul = False
                result.append('<ol>')
                in_ol = True
            item = re.sub(r'^\d+\. ', '', stripped)
            item = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item)
            item = re.sub(r'`([^`]+)`', r'<code class="inline-code">\1</code>', item)
            result.append(f'<li>{item}</li>')
        else:
            if in_ul:
                result.append('</ul>')
                in_ul = False
            if in_ol:
                result.append('</ol>')
                in_ol = False
            
            if stripped and not stripped.startswith('<h') and not stripped.startswith('<pre'):
                processed = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', stripped)
                processed = re.sub(r'`([^`]+)`', r'<code class="inline-code">\1</code>', processed)
                if processed:
                    result.append(f'<p>{processed}</p>')
            elif stripped.startswith('<pre') or stripped.startswith('<h'):
                result.append(stripped)
    
    if in_ul:
        result.append('</ul>')
    if in_ol:
        result.append('</ol>')
    
    html = '\n'.join(result)
    
    # Restore mermaid blocks
    for idx, mermaid_block in enumerate(mermaid_blocks):
        mermaid_content = re.search(r'```mermaid\n(.*?)\n```', mermaid_block, re.DOTALL)
        if mermaid_content:
            mermaid_code = mermaid_content.group(1)
            html = html.replace(f'MERMAID_BLOCK_{idx}', 
                              f'<div class="diagram-container"><div class="mermaid">\n{mermaid_code}\n</div></div>')
    
    # Clean up
    html = re.sub(r'<p>\s*</p>', '', html)
    html = re.sub(r'\n{3,}', '\n\n', html)
    
    return html

def create_chapter_html(chapter, content_text):
    """Create HTML for a chapter"""
    prev_num = chapter['num'] - 1
    next_num = chapter['num'] + 1
    
    nav_html = f'''
    <!-- Mobile Menu Toggle -->
    <button class="mobile-menu-toggle" onclick="toggleMobileMenu()" aria-label="Toggle menu">☰</button>

    <nav class="nav-sidebar" id="nav-sidebar">
        <ul>
            <li><a href="comprehensive_index.html">🏠 Home</a></li>
            <li><a href="comprehensive_chapter1.html">🤖 Chapter 1: Introduction to AI</a></li>
            <li><a href="comprehensive_chapter2.html">📊 Chapter 2: Machine Learning</a></li>
            <li><a href="comprehensive_chapter3.html">🧠 Chapter 3: Deep Learning</a></li>
            <li><a href="comprehensive_chapter4.html">🔗 Chapter 4: Neural Networks</a></li>
            <li><a href="comprehensive_chapter5.html">💬 Chapter 5: NLP Evolution</a></li>
            <li><a href="comprehensive_chapter6.html">⚡ Chapter 6: Transformers</a></li>
            <li><a href="comprehensive_chapter7.html">🎓 Chapter 7: LLM Training</a></li>
            <li><a href="comprehensive_chapter8.html">🏗️ Chapter 8: LLM Architecture</a></li>
            <li><a href="comprehensive_chapter9.html">🔄 Chapter 9: Query Processing</a></li>
            <li><a href="comprehensive_chapter10.html">👁️ Chapter 10: Attention</a></li>
            <li><a href="comprehensive_chapter11.html">📚 Chapter 11: Training Data</a></li>
            <li><a href="comprehensive_chapter12.html">🎯 Chapter 12: Fine-tuning</a></li>
            <li><a href="comprehensive_chapter13.html">⚙️ Chapter 13: Inference</a></li>
            <li><a href="comprehensive_chapter14.html">📈 Chapter 14: Evolution</a></li>
            <li><a href="comprehensive_chapter15.html">🚀 Chapter 15: Applications</a></li>
        </ul>
    </nav>'''
    
    chapter_nav = '<div class="chapter-nav">'
    if prev_num >= 1:
        chapter_nav += f'<a href="comprehensive_chapter{prev_num}.html">← Previous Chapter</a>'
    else:
        chapter_nav += '<span></span>'
    chapter_nav += f'<a href="comprehensive_index.html">🏠 Home</a>'
    if next_num <= 15:
        chapter_nav += f'<a href="comprehensive_chapter{next_num}.html">Next Chapter →</a>'
    else:
        chapter_nav += '<span></span>'
    chapter_nav += '</div>'
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{chapter['title']} - Comprehensive LLM Guide</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>{chapter['icon']} {chapter['title']}</h1>
            <p class="subtitle">Chapter {chapter['num']} of 15 - Comprehensive Guide</p>
        </header>

        {nav_html}

        <main class="main-content">
            <div class="chapter-header">
                <h1>{chapter['icon']} Chapter {chapter['num']}: {chapter['title']}</h1>
                <p>Comprehensive Learning Guide - Detailed Presentation Material</p>
            </div>

            {chapter_nav}

            <div class="section">
                {content_text}
            </div>

            {chapter_nav}
        </main>

        <footer class="footer">
            <p>© 2024 NL2SQL Project - Comprehensive LLM Learning Guide</p>
            <p>Chapter {chapter['num']} of 15</p>
        </footer>
    </div>

    <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'default',
            flowchart: {{ useMaxWidth: true, htmlLabels: true }}
        }});
        
        // Mobile menu toggle
        function toggleMobileMenu() {{
            const nav = document.getElementById('nav-sidebar');
            if (nav) nav.classList.toggle('open');
        }}

        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {{
            const nav = document.getElementById('nav-sidebar');
            const toggle = document.querySelector('.mobile-menu-toggle');
            
            if (nav && toggle && 
                !nav.contains(event.target) && 
                !toggle.contains(event.target) &&
                nav.classList.contains('open')) {{
                nav.classList.remove('open');
            }}
        }});

        // Close mobile menu when clicking a link
        document.querySelectorAll('.nav-sidebar a').forEach(link => {{
            link.addEventListener('click', function() {{
                const nav = document.getElementById('nav-sidebar');
                if (nav && window.innerWidth <= 768) {{
                    nav.classList.remove('open');
                }}
            }});
        }});
    </script>
</body>
</html>'''
    
    return html

# Generate all chapter files
for chapter in chapters:
    chapter_content = extract_chapter_content(content, chapter['start'], chapter['end'])
    if chapter_content:
        # Remove the main header
        chapter_content = re.sub(r'^## \d+\. .*?\n', '', chapter_content, count=1)
        
        # Convert markdown to HTML
        html_content = markdown_to_html(chapter_content)
        full_html = create_chapter_html(chapter, html_content)
        
        filename = f"comprehensive_chapter{chapter['num']}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"Generated {filename}")

print("All comprehensive chapters generated!")

