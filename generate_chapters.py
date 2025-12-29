#!/usr/bin/env python3
"""
Script to generate HTML chapter files from the markdown guide
"""

import re
import os

# Read the markdown file
with open('../LLM_Deep_Learning_Guide.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Chapter definitions with their content ranges
chapters = [
    {
        'num': 1,
        'title': 'Introduction to Large Language Models',
        'icon': '📖',
        'start': '## 1. Introduction to Large Language Models',
        'end': '## 2. How LLMs Work'
    },
    {
        'num': 2,
        'title': 'How LLMs Work: The Transformer Architecture',
        'icon': '⚙️',
        'start': '## 2. How LLMs Work: The Transformer Architecture',
        'end': '## 3. Tokenization'
    },
    {
        'num': 3,
        'title': 'Tokenization: Converting Text to Numbers',
        'icon': '🔢',
        'start': '## 3. Tokenization: Converting Text to Numbers',
        'end': '## 4. Attention Mechanism'
    },
    {
        'num': 4,
        'title': 'Attention Mechanism: The Core Innovation',
        'icon': '👁️',
        'start': '## 4. Attention Mechanism: The Core Innovation',
        'end': '## 5. Training LLMs'
    },
    {
        'num': 5,
        'title': 'Training LLMs: Pre-training and Fine-tuning',
        'icon': '🎓',
        'start': '## 5. Training LLMs: Pre-training and Fine-tuning',
        'end': '## 6. How LLMs Generate Text'
    },
    {
        'num': 6,
        'title': 'How LLMs Generate Text',
        'icon': '✨',
        'start': '## 6. How LLMs Generate Text',
        'end': '## 7. Why LLMs Succeed'
    },
    {
        'num': 7,
        'title': 'Why LLMs Succeed',
        'icon': '✅',
        'start': '## 7. Why LLMs Succeed',
        'end': '## 8. Why LLMs Fail'
    },
    {
        'num': 8,
        'title': 'Why LLMs Fail',
        'icon': '❌',
        'start': '## 8. Why LLMs Fail',
        'end': '## 9. NL2SQL'
    },
    {
        'num': 9,
        'title': 'NL2SQL: Natural Language to SQL',
        'icon': '💬',
        'start': '## 9. NL2SQL: Natural Language to SQL',
        'end': '## 10. Your NL2SQL System Architecture'
    },
    {
        'num': 10,
        'title': 'Your NL2SQL System Architecture',
        'icon': '🏗️',
        'start': '## 10. Your NL2SQL System Architecture',
        'end': '## 11. Practical Examples'
    },
    {
        'num': 11,
        'title': 'Practical Examples and Case Studies',
        'icon': '📚',
        'start': '## 11. Practical Examples and Case Studies',
        'end': '## Summary: Key Takeaways'
    }
]

def extract_chapter_content(content, start_marker, end_marker):
    """Extract content between two markers"""
    start_idx = content.find(start_marker)
    if start_idx == -1:
        return None
    
    end_idx = content.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        # Last chapter, get until end
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
        # Escape HTML in code
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
    <nav class="nav-sidebar">
        <ul>
            <li><a href="index.html">🏠 Home</a></li>
            <li><a href="chapter1.html">📖 Chapter 1: Introduction to LLMs</a></li>
            <li><a href="chapter2.html">⚙️ Chapter 2: Transformer Architecture</a></li>
            <li><a href="chapter3.html">🔢 Chapter 3: Tokenization</a></li>
            <li><a href="chapter4.html">👁️ Chapter 4: Attention Mechanism</a></li>
            <li><a href="chapter5.html">🎓 Chapter 5: Training LLMs</a></li>
            <li><a href="chapter6.html">✨ Chapter 6: Text Generation</a></li>
            <li><a href="chapter7.html">✅ Chapter 7: Why LLMs Succeed</a></li>
            <li><a href="chapter8.html">❌ Chapter 8: Why LLMs Fail</a></li>
            <li><a href="chapter9.html">💬 Chapter 9: NL2SQL Overview</a></li>
            <li><a href="chapter10.html">🏗️ Chapter 10: Your NL2SQL System</a></li>
            <li><a href="chapter11.html">📚 Chapter 11: Examples & Case Studies</a></li>
        </ul>
    </nav>'''
    
    chapter_nav = '<div class="chapter-nav">'
    if prev_num >= 1:
        chapter_nav += f'<a href="chapter{prev_num}.html">← Previous Chapter</a>'
    else:
        chapter_nav += '<span></span>'
    chapter_nav += f'<a href="index.html">🏠 Home</a>'
    if next_num <= 11:
        chapter_nav += f'<a href="chapter{next_num}.html">Next Chapter →</a>'
    else:
        chapter_nav += '<span></span>'
    chapter_nav += '</div>'
    
    # Process mermaid diagrams
    content_html = content_text
    # Keep mermaid blocks as-is (they'll be rendered by mermaid.js)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{chapter['title']} - LLM Learning Guide</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>{chapter['icon']} {chapter['title']}</h1>
            <p class="subtitle">Chapter {chapter['num']} of 11</p>
        </header>

        {nav_html}

        <main class="main-content">
            <div class="chapter-header">
                <h1>{chapter['icon']} {chapter['title']}</h1>
                <p>Chapter {chapter['num']} of 11 - Complete Learning Guide</p>
            </div>

            {chapter_nav}

            <div class="section">
                {content_html}
            </div>

            {chapter_nav}
        </main>

        <footer class="footer">
            <p>© 2024 NL2SQL Project - Complete LLM Learning Guide</p>
            <p>Chapter {chapter['num']} of 11</p>
        </footer>
    </div>

    <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'default',
            flowchart: {{ useMaxWidth: true, htmlLabels: true }}
        }});
    </script>
</body>
</html>'''
    
    return html

# Generate all chapter files
for chapter in chapters:
    chapter_content = extract_chapter_content(content, chapter['start'], chapter['end'])
    if chapter_content:
        # Remove the main header (## 1. Title)
        chapter_content = re.sub(r'^## \d+\. .*?\n', '', chapter_content, count=1)
        
        # Convert markdown to HTML
        html_content = markdown_to_html(chapter_content)
        full_html = create_chapter_html(chapter, html_content)
        
        filename = f"chapter{chapter['num']}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"Generated {filename}")

print("All chapters generated!")

