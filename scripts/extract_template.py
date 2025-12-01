from docx import Document
import os
p = r'd:/thesis work/densnet-thesis/thesis writing/Master Thesis Proposal.docx'
if not os.path.exists(p):
    raise SystemExit('file not found: '+p)

doc = Document(p)
print('PARA_INDEX\tSTYLE\tTEXT')
for i, para in enumerate(doc.paragraphs):
    text = para.text.strip()
    if not text:
        continue
    style = para.style.name if para.style is not None else 'None'
    print(f"{i}\t{style}\t{text}")
