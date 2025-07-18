## Convert PDF and .docx to .txt files

import os
from pathlib import Path
from pdfminer.high_level import extract_text
from docx import Document

def process_files(folder_path: str = 'HRPP') -> None:
    # check if folder exists
    if not os.path.exists(folder_path):
        print(f"{folder_path} does not exist")
        return
    
    # extract all .pdf & .docx
    folder_path = Path(folder_path)
    files = list(folder_path.rglob("*.pdf")) + list(folder_path.rglob("*.docx"))

    # convert & save each file
    for file_path in files:
        print(f"Opening {file_path}")
        
        try:
            if '.pdf' in str(file_path):  # .pdf
                text = extract_text(file_path)
            else:  # .docx
                full_text = []

                doc = Document(file_path)
                for para in doc.paragraphs:
                    full_text.append(para.text)

                text = '\n'.join(full_text)

            # save file as .txt
            folder = 'HRPP_text'
            os.makedirs(folder, exist_ok=True)
            output_path = os.path.join(folder, f'{Path(file_path).stem}.txt')

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

if __name__ == "__main__":
    process_files()