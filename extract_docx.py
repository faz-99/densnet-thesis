from docx import Document
import sys

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []
        
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading document: {str(e)}"

if __name__ == "__main__":
    docx_path = r"d:\thesis work\densnet-thesis\thesis writing\Master Thesis Proposal.docx"
    text = extract_text_from_docx(docx_path)
    
    # Save to text file
    with open(r"d:\thesis work\densnet-thesis\thesis_format.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    print("Text extracted and saved to thesis_format.txt")