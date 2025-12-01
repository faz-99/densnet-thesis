from docx import Document

def extract_template_structure(file_path):
    try:
        doc = Document(file_path)
        
        content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                style_name = paragraph.style.name if paragraph.style else "Normal"
                content.append(f"[{style_name}] {paragraph.text}")
        
        return '\n'.join(content)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    template_path = r"d:\thesis work\densnet-thesis\thesis writing\template.docx"
    content = extract_template_structure(template_path)
    
    with open(r"d:\thesis work\densnet-thesis\template_structure.txt", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("Template structure extracted")