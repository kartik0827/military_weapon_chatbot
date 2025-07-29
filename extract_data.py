import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    pdf_path = r"C:\Users\Kartik\Desktop\chatbot\data\military_weapons_dataset.pdf"
    text = extract_text_from_pdf(pdf_path)
    with open("extracted/military_weapons.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("✅ Text extracted to extracted/military_weapons.txt")
