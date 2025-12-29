#!/usr/bin/env python3
"""
Add mobile menu toggle to all chapter HTML files
"""

import os
import re

def update_chapter_file(filename):
    """Add mobile menu toggle to a chapter file"""
    if not os.path.exists(filename):
        return False
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already updated
    if 'mobile-menu-toggle' in content:
        return False
    
    # Add mobile menu toggle button after container div
    content = re.sub(
        r'(<div class="container">)',
        r'\1\n        <!-- Mobile Menu Toggle -->\n        <button class="mobile-menu-toggle" onclick="toggleMobileMenu()" aria-label="Toggle menu">☰</button>',
        content
    )
    
    # Update nav sidebar to have id
    content = re.sub(
        r'(<nav class="nav-sidebar">)',
        r'<nav class="nav-sidebar" id="nav-sidebar">',
        content
    )
    
    # Add mobile menu JavaScript before closing body tag
    mobile_js = '''
        // Mobile menu toggle
        function toggleMobileMenu() {
            const nav = document.getElementById('nav-sidebar');
            nav.classList.toggle('open');
        }

        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {
            const nav = document.getElementById('nav-sidebar');
            const toggle = document.querySelector('.mobile-menu-toggle');
            
            if (nav && toggle && 
                !nav.contains(event.target) && 
                !toggle.contains(event.target) &&
                nav.classList.contains('open')) {
                nav.classList.remove('open');
            }
        });

        // Close mobile menu when clicking a link
        document.querySelectorAll('.nav-sidebar a').forEach(link => {
            link.addEventListener('click', function() {
                const nav = document.getElementById('nav-sidebar');
                if (nav && window.innerWidth <= 768) {
                    nav.classList.remove('open');
                }
            });
        });
'''
    
    # Insert before closing script tag or before closing body tag
    if '</script>' in content:
        content = re.sub(
            r'(</script>)',
            mobile_js + r'\1',
            content,
            count=1
        )
    else:
        content = re.sub(
            r'(</body>)',
            r'<script>' + mobile_js + '</script>\n    \1',
            content
        )
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

# Update all chapter files
chapters = [f'chapter{i}.html' for i in range(1, 12)]

for chapter in chapters:
    if update_chapter_file(chapter):
        print(f"Updated {chapter}")
    else:
        print(f"{chapter} - already updated or not found")

print("Done!")

