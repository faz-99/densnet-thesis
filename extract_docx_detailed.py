from docx import Document
from docx.shared import Inches

def extract_detailed_structure(file_path):
    try:
        doc = Document(file_path)
        
        structure = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                # Get style information
                style_name = paragraph.style.name if paragraph.style else "Normal"
                
                # Check if it's a heading
                if 'Heading' in style_name or paragraph.text.isupper() or len(paragraph.text) < 100:
                    structure.append(f"[{style_name}] {paragraph.text}")
                else:
                    structure.append(paragraph.text)
        
        return '\n'.join(structure)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    docx_path = r"d:\thesis work\densnet-thesis\thesis writing\Master Thesis Proposal.docx"
    structure = extract_detailed_structure(docx_path)
    
    with open(r"d:\thesis work\densnet-thesis\thesis_structure.txt", "w", encoding="utf-8") as f:
        f.write(structure)
    
    print("Structure extracted to thesis_structure.txt")