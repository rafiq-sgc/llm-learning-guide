#!/usr/bin/env python3
"""
Improved markdown to HTML converter for chapter content
"""

import re

def convert_markdown_to_html(markdown_text):
    """Convert markdown text to HTML"""
    html = markdown_text
    
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
        return f'<pre><code>{code}</code></pre>'
    
    html = re.sub(r'```(\w+)?\n(.*?)```', code_block, html, flags=re.DOTALL)
    
    # Convert headers
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # Convert bold
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    
    # Convert inline code
    html = re.sub(r'`([^`\n]+)`', r'<code class="inline-code">\1</code>', html)
    
    # Convert lists - handle both unordered and ordered
    lines = html.split('\n')
    result = []
    in_ul = False
    in_ol = False
    
    for line in lines:
        stripped = line.strip()
        
        # Unordered list
        if re.match(r'^- ', stripped):
            if not in_ul:
                if in_ol:
                    result.append('</ol>')
                    in_ol = False
                result.append('<ul>')
                in_ul = True
            item = re.sub(r'^- ', '', stripped)
            # Process bold and inline code in list items
            item = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item)
            item = re.sub(r'`([^`]+)`', r'<code class="inline-code">\1</code>', item)
            result.append(f'<li>{item}</li>')
        
        # Ordered list
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
        
        # Regular line
        else:
            if in_ul:
                result.append('</ul>')
                in_ul = False
            if in_ol:
                result.append('</ol>')
                in_ol = False
            
            if stripped:
                # Skip if it's a header (already converted)
                if not stripped.startswith('<h'):
                    # Process bold and code
                    processed = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', stripped)
                    processed = re.sub(r'`([^`]+)`', r'<code class="inline-code">\1</code>', processed)
                    result.append(f'<p>{processed}</p>')
    
    if in_ul:
        result.append('</ul>')
    if in_ol:
        result.append('</ol>')
    
    html = '\n'.join(result)
    
    # Restore mermaid blocks
    for idx, mermaid_block in enumerate(mermaid_blocks):
        # Extract mermaid content
        mermaid_content = re.search(r'```mermaid\n(.*?)\n```', mermaid_block, re.DOTALL)
        if mermaid_content:
            mermaid_code = mermaid_content.group(1)
            html = html.replace(f'MERMAID_BLOCK_{idx}', 
                              f'<div class="diagram-container"><div class="mermaid">\n{mermaid_code}\n</div></div>')
    
    # Clean up empty paragraphs
    html = re.sub(r'<p>\s*</p>', '', html)
    
    # Clean up multiple newlines
    html = re.sub(r'\n{3,}', '\n\n', html)
    
    return html

if __name__ == '__main__':
    # Test
    test_md = """### Test Header

This is a paragraph with **bold** text and `inline code`.

- List item 1
- List item 2 with **bold**

```python
print("Hello")
```

```mermaid
graph TD
    A --> B
```
"""
    print(convert_markdown_to_html(test_md))

