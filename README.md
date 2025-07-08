import tkinter as tk
from tkinter import filedialog, Text, messagebox, Label, Button
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
import os
import re
import torch
import threading
from torchvision import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pytesseract
import easyocr

class UnifiedOCRApp:
    def _init_(self):
        self.root = tk.Tk()
        self.root.title("OCR Scanner")
        self.root.geometry("1000x800")
        self.init_model()
        self.init_gui()

    def init_model(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(self.device)
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def init_gui(self):
        Button(self.root, text="Upload Image", command=self.load_image).pack(pady=10)
        Button(self.root, text="Scan via Webcam", command=self.scan).pack(pady=5)
        self.image_panel = Label(self.root)
        self.image_panel.pack()
        self.output_text = Text(self.root, height=15, width=120)
        self.output_text.pack(pady=10)
        Button(self.root, text="Save Text", command=self.save_text).pack()
        Button(self.root, text="Clear", command=self.clear_all).pack(pady=5)
        self.status_label = Label(self.root, text="Status: Ready", fg="green")
        self.status_label.pack(pady=5)

    def enhance_image(self, img):
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        return img

    def preprocess_for_rotation(self, img):
        arr = np.array(img)
        binary = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(binary)
        coords = np.column_stack(np.where(denoised > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle += 90
        h, w = denoised.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def recognize_text(self, pil_image):
        texts = []
        try:
            enhanced = self.enhance_image(pil_image)
            pre_img = self.preprocess_for_rotation(enhanced)
            for cfg in ['--oem 3 --psm 6', '--oem 3 --psm 4']:
                text = pytesseract.image_to_string(pre_img, config=f'{cfg} -c tessedit_char_whitelist="A-Za-z0-9.,\'\":;()- "')
                if text.strip():
                    texts.append("[Tesseract] " + text.strip())
        except:
            pass
        try:
            easy_text = self.reader.readtext(np.array(pil_image))
            if easy_text:
                texts.append("[EasyOCR] " + " ".join([t[1] for t in easy_text]))
        except:
            pass
        try:
            resized = pil_image.convert("RGB").resize((1024, 1024))
            pixel_values = self.processor(images=resized, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if len(text.strip()) > 3:
                texts.append("[TrOCR] " + text.strip())
        except:
            pass
        return self.clean_text(" | ".join(texts)) if texts else "No readable text found."

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        return text.strip()

    def display_image(self, img):
        img_resized = img.resize((600, 400))
        photo = ImageTk.PhotoImage(img_resized)
        self.image_panel.config(image=photo)
        self.image_panel.image = photo

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.status_label.config(text="Status: Processing...", fg="blue")
            img = Image.open(path).convert("RGB")
            self.display_image(img)
            text = self.recognize_text(img)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, text)
            self.status_label.config(text="Status: Done", fg="green")

    def save_text(self):
        text = self.output_text.get(1.0, tk.END).strip()
        if text:
            path = filedialog.asksaveasfilename(defaultextension=".txt")
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)

    def clear_all(self):
        self.output_text.delete(1.0, tk.END)
        self.image_panel.config(image='')
        self.image_panel.image = None
        self.status_label.config(text="Status: Ready", fg="green")

    def scan(self):
        self.cam_window = tk.Toplevel(self.root)
        self.cam_window.title("Live Webcam")
        self.cam_window.geometry("700x550")

        self.video_label = Label(self.cam_window)
        self.video_label.pack()

        self.capture_button = Button(self.cam_window, text="📸 Capture Photo", command=self.capture_frame, bg="lightblue", font=("Arial", 12))
        self.capture_button.pack(pady=10)

        self.snapshot_label = Label(self.cam_window, text="", font=("Arial", 10), fg="green")
        self.snapshot_label.pack()

        self.stop_event = threading.Event()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Webcam not available.")
            self.cam_window.destroy()
            return

        self.show_frame()
        self.cam_window.protocol("WM_DELETE_WINDOW", self.close_camera)

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        if not self.stop_event.is_set():
            self.cam_window.after(10, self.show_frame)

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            path = "captured_from_webcam.png"
            cv2.imwrite(path, frame)
            self.snapshot_label.config(text="✅ Photo Captured! Processing...")
            self.root.after(500, lambda: [self.close_camera(), self.load_image_from_path(path)])

    def close_camera(self):
        self.stop_event.set()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.cam_window.destroy()

    def load_image_from_path(self, path):
        img = Image.open(path).convert("RGB")
        self.display_image(img)
        text = self.recognize_text(img)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.status_label.config(text="Status: Done", fg="green")

    def run(self):
        self.root.mainloop()

if _name_ == "_main_":
    UnifiedOCRApp().run()
