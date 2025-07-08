# Past_Ink-
 Unified OCR Scanner (Handwritten & Printed Text)
A powerful, Python-based desktop application for extracting handwritten and printed text from images or live webcam input using multiple OCR engines—all through a simple Tkinter GUI.

🔍 Features
📸 Image Upload & Webcam Capture
Easily scan documents from files or take real-time photos using your webcam.

🧠 Multi-Engine OCR Support
Utilizes three OCR models:

Tesseract OCR: Traditional open-source engine

EasyOCR: Lightweight and multilingual

TrOCR (Transformers OCR): Microsoft’s transformer-based model for handwritten text

🧹 Smart Preprocessing with OpenCV & PIL
Enhances images using contrast, sharpening, grayscale, and skew correction for improved recognition.

💾 Save Extracted Text
Store recognized text to .txt files with a single click.

🧼 Text Cleaning
Automatically removes extra whitespace and formats punctuation for better readability.

🛠️ Built With
Python

Tkinter — GUI Framework

OpenCV — Image preprocessing

Pytesseract — Wrapper for Tesseract OCR

EasyOCR — Deep learning OCR library

Transformers — For TrOCR model from HuggingFace

PIL (Pillow) — Image manipulation

📦 How to Run
Install dependencies:

bash
Copy
Edit
pip install torch torchvision transformers pytesseract easyocr opencv-python pillow
Make sure Tesseract OCR is installed and configured (Windows default path: C:\Program Files\Tesseract-OCR\tesseract.exe).

Run the application:

bash
Copy
Edit
python your_script.py
📸 Sample Use Cases
Digitizing handwritten notes

Extracting text from scanned documents

Archiving old records

OCR for academic or research papers
