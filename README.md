# LLM Learning Guide - HTML Version

A beautiful, interactive HTML version of the Complete LLM Learning Guide with visualizations, diagrams, and easy navigation.

## 📁 Structure

```
llm-learning-guide/
├── index.html              # Home page with table of contents
├── chapter1.html           # Introduction to Large Language Models
├── chapter2.html           # How LLMs Work: The Transformer Architecture
├── chapter3.html           # Tokenization: Converting Text to Numbers
├── chapter4.html           # Attention Mechanism: The Core Innovation
├── chapter5.html           # Training LLMs: Pre-training and Fine-tuning
├── chapter6.html           # How LLMs Generate Text
├── chapter7.html           # Why LLMs Succeed
├── chapter8.html           # Why LLMs Fail
├── chapter9.html           # NL2SQL: Natural Language to SQL
├── chapter10.html          # Your NL2SQL System Architecture
├── chapter11.html          # Practical Examples and Case Studies
├── styles.css              # Shared stylesheet
├── generate_chapters.py    # Script to regenerate chapters from markdown
└── README.md               # This file
```

## 🚀 Usage

### Viewing the Guide

1. **Open in Browser**: Simply open `index.html` in your web browser
   ```bash
   # From the llm-learning-guide directory
   # On Linux/Mac:
   open index.html
   # Or use any web browser to open the file
   ```

2. **Local Server** (Recommended for best experience):
   ```bash
   # Python 3
   python3 -m http.server 8000
   
   # Then open: http://localhost:8000
   ```

3. **VS Code Live Server**: If using VS Code, right-click `index.html` and select "Open with Live Server"

### Navigation

- **Home Page**: Click "🏠 Home" in the sidebar or navigate to `index.html`
- **Chapters**: Click any chapter link in the sidebar or table of contents
- **Next/Previous**: Use the navigation buttons at the top and bottom of each chapter
- **Sidebar**: Always visible on the left for quick navigation

## 🎨 Features

### Visual Design
- **Modern UI**: Clean, professional design with gradient headers
- **Responsive**: Works on desktop, tablet, and mobile devices
- **Color-Coded**: Different colors for different types of content
- **Smooth Navigation**: Easy navigation between chapters

### Interactive Elements
- **Mermaid Diagrams**: All diagrams are interactive and rendered with Mermaid.js
- **Code Highlighting**: Code blocks are properly formatted
- **Collapsible Sections**: Easy to read and navigate
- **Visual Examples**: Color-coded example boxes (success, warning, error)

### Content Organization
- **11 Comprehensive Chapters**: Complete coverage of LLM topics
- **Real Examples**: Examples from your actual NL2SQL system
- **Visual Diagrams**: Flowcharts, sequence diagrams, and architecture diagrams
- **Code Snippets**: Real code examples from your codebase

## 📝 Regenerating Chapters

If you update the source markdown file (`../LLM_Deep_Learning_Guide.md`), you can regenerate all HTML chapters:

```bash
cd llm-learning-guide
python3 generate_chapters.py
```

This will:
1. Read the markdown file
2. Extract each chapter
3. Convert markdown to HTML
4. Generate complete HTML files with navigation

## 🎯 Learning Path

### Recommended Order
1. Start with **Chapter 1** (Introduction)
2. Progress through **Chapters 2-6** (Core concepts)
3. Study **Chapters 7-8** (Success and failure modes)
4. Focus on **Chapters 9-11** (NL2SQL and your system)

### Quick Access
- **New to LLMs?** → Start at Chapter 1
- **Want to understand your system?** → Jump to Chapter 10
- **Need examples?** → Go to Chapter 11
- **Understanding failures?** → Check Chapter 8

## 🔧 Customization

### Styling
Edit `styles.css` to customize:
- Colors (CSS variables in `:root`)
- Fonts
- Layout
- Spacing

### Content
1. Edit the source markdown: `../LLM_Deep_Learning_Guide.md`
2. Regenerate chapters: `python3 generate_chapters.py`

## 📱 Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Mobile browsers

## 🎓 Tips for Learning

1. **Read Sequentially**: Chapters build on each other
2. **Study Diagrams**: Visual diagrams help understand complex concepts
3. **Try Examples**: Run the code examples in your system
4. **Take Notes**: Use browser bookmarks or print pages
5. **Review**: Revisit chapters as needed

## 📄 Printing

The guide is print-friendly:
- Use browser's Print function (Ctrl+P / Cmd+P)
- Navigation and sidebars are hidden in print view
- Content is optimized for paper

## 🐛 Troubleshooting

### Diagrams Not Showing
- Ensure you have internet connection (Mermaid.js loads from CDN)
- Check browser console for errors
- Try refreshing the page

### Styling Issues
- Clear browser cache
- Ensure `styles.css` is in the same directory
- Check file paths are correct

### Navigation Not Working
- Ensure all HTML files are in the same directory
- Check that file names match exactly (case-sensitive on Linux/Mac)

## 📚 Additional Resources

- Source Markdown: `../LLM_Deep_Learning_Guide.md`
- Presentation: `../Presentation_Behind_Scenes_LLMs.md`
- Diagrams: `../presentation_diagrams.html`

## 💡 Feedback

This guide is designed to help you:
- Understand LLMs deeply
- Learn how your NL2SQL system works
- Prepare for presentations
- Share knowledge with your team

Enjoy learning! 🚀

