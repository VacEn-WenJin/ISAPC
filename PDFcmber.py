import os
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog, messagebox, Frame, Label, Button
import numpy as np
from PIL import Image

def create_pdf_grid(pdf_paths, output_path="combined_grid.png"):
    if len(pdf_paths) != 9:
        raise ValueError("Exactly 9 PDF files are required")
    
    # Extract first page of each PDF as an image
    images = []
    max_width, max_height = 0, 0
    
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)  # First page
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            
            # Track maximum dimensions
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
            
            doc.close()
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return None
    
    # Create a blank canvas for the 3x3 grid
    grid_img = Image.new('RGB', (max_width * 3, max_height * 3), (255, 255, 255))
    
    # Place each image in the grid
    for i, img in enumerate(images):
        row = i // 3
        col = i % 3
        # Center the image in its cell
        x = col * max_width + (max_width - img.width) // 2
        y = row * max_height + (max_height - img.height) // 2
        grid_img.paste(img, (x, y))
    
    # Save the grid image
    grid_img.save(output_path)
    return output_path

class PDFGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Grid Creator")
        self.pdf_files = []
        
        # Configure main window
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")
        
        # Create UI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="PDF to 3×3 Grid Converter", 
                              font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(self.root, 
                               text="Select 9 PDF files to create a 3×3 grid", 
                               bg="#f0f0f0")
        instructions.pack(pady=5)
        
        # File selection area
        self.selection_frame = tk.Frame(self.root, bg="#e0e0e0", bd=2, relief="groove")
        self.selection_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        select_btn = tk.Button(self.selection_frame, text="Select PDFs", command=self.select_pdfs)
        select_btn.pack(pady=20)
        
        # File list
        self.file_list_frame = tk.Frame(self.selection_frame, bg="#e0e0e0")
        self.file_list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.file_list_label = tk.Label(self.file_list_frame, 
                                       text="No files selected", 
                                       bg="#e0e0e0",
                                       justify=tk.LEFT,
                                       wraplength=500)
        self.file_list_label.pack(anchor="w")
        
        # Status area
        self.status_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.status_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_label = tk.Label(self.status_frame, 
                                    text="0/9 files selected", 
                                    bg="#f0f0f0")
        self.status_label.pack(side="left")
        
        # Process button
        self.process_btn = tk.Button(self.root, text="Create Grid", 
                                    command=self.process_pdfs, 
                                    state="disabled")
        self.process_btn.pack(pady=15)
        
        # Clear button
        clear_btn = tk.Button(self.root, text="Clear Selection", 
                             command=self.clear_selection)
        clear_btn.pack(pady=5)
    
    def select_pdfs(self):
        new_files = filedialog.askopenfilenames(
            title="Select PDF files",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if new_files:
            # Add new files to the existing selection, up to 9 files
            remaining_slots = 9 - len(self.pdf_files)
            self.pdf_files.extend(list(new_files)[:remaining_slots])
            # Keep only the first 9 files if more than 9 are selected
            self.pdf_files = self.pdf_files[:9]
            self.update_status()
    
    def clear_selection(self):
        self.pdf_files = []
        self.update_status()
    
    def update_status(self):
        count = len(self.pdf_files)
        self.status_label.config(text=f"{count}/9 files selected")
        
        # Update file list display
        if count > 0:
            file_list_text = "\n".join([os.path.basename(f) for f in self.pdf_files])
            self.file_list_label.config(text=file_list_text)
        else:
            self.file_list_label.config(text="No files selected")
        
        if count == 9:
            self.process_btn.config(state="normal")
        else:
            self.process_btn.config(state="disabled")
    
    def process_pdfs(self):
        if len(self.pdf_files) != 9:
            messagebox.showwarning("Warning", "Please select exactly 9 PDF files.")
            return
        
        output_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
        )
        
        if output_path:
            try:
                result = create_pdf_grid(self.pdf_files, output_path)
                if result:
                    messagebox.showinfo("Success", f"Grid image saved to:\n{output_path}")
                else:
                    messagebox.showerror("Error", "Failed to create grid image")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFGridApp(root)
    root.mainloop()