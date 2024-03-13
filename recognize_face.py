import shutil


import tkinter as tk
import os
import cv2
import face_recognition
import json
import numpy as np
from PIL import Image, ImageTk
from scipy.spatial import distance
from tkinter import ttk, filedialog, messagebox
import threading

# Function to load saved encodings
def load_encodings():
    with open('encodings.json', 'r') as file:
        data = json.load(file)
    return [{**item, "encoding": np.array(item["encoding"])} for item in data]

# Function to recognize faces
def recognize_face(face_encoding, stored_encodings, tolerance=0.7):
    distances = [(item["name"], distance.euclidean(face_encoding, item["encoding"])) for item in stored_encodings]
    distances.sort(key=lambda x: x[1])
    if distances and distances[0][1] <= tolerance:
        return distances[0][0]
    return "Unknown"

# Function to process folder
# def process_folder(source_folder, target_folder, stored_encodings):
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
#     for subdir, dirs, files in os.walk(source_folder):
#         for file in files:
#             file_path = os.path.join(subdir, file)
#             with Image.open(file_path) as img:
#                 base_width = 800
#                 w_percent = (base_width / float(img.size[0]))
#                 h_size = int((float(img.size[1]) * float(w_percent)))
#                 img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
#                 image = np.array(img)
#                 image = image[:, :, :3]
#             encodings = face_recognition.face_encodings(image, model="cnn")
#             recognized_names = set()
#             for encoding in encodings:
#                 recognized_name = recognize_face(encoding, stored_encodings)
#                 recognized_names.add(recognized_name)
#             for recognized_name in recognized_names:
#                 target_directory = os.path.join(target_folder, recognized_name)
#                 if not os.path.exists(target_directory):
#                     os.makedirs(target_directory)
#                 target_file_path = os.path.join(target_directory, file)
#                 shutil.copy(file_path, target_file_path)
#                 print(f"Copied: {file_path} to {target_file_path}")

def process_folder(source_folder, target_folder, stored_encodings, update_progress_callback, completion_callback=None):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    total_files = sum([len(files) for r, d, files in os.walk(source_folder)])
    processed_files = 0

    for subdir, dirs, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    base_width = 800
                    w_percent = (base_width / float(img.size[0]))
                    h_size = int((float(img.size[1]) * float(w_percent)))
                    img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
                    image = np.array(img)
                    image = image[:, :, :3]

                encodings = face_recognition.face_encodings(image, model="cnn")

                recognized_names = set()
                for encoding in encodings:
                    recognized_name = recognize_face(encoding, stored_encodings)
                    recognized_names.add(recognized_name)

                for recognized_name in recognized_names:
                    target_directory = os.path.join(target_folder, recognized_name)
                    if not os.path.exists(target_directory):
                        os.makedirs(target_directory)
                    target_file_path = os.path.join(target_directory, file)
                    shutil.copy(file_path, target_file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
            
            processed_files += 1
            if update_progress_callback:
                update_progress_callback(processed_files, total_files)
            
    if completion_callback:
        completion_callback()


class FaceSorterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Sorter")
        self.geometry("400x200")

        # Set up the UI
        self.create_widgets()

        # Load stored encodings
        self.stored_encodings = load_encodings()

    def create_widgets(self):
        style = ttk.Style(self)
        style.theme_use('clam')  # 'clam' is a ttk theme that looks more modern than default

        self.source_label = ttk.Label(self, text="Source Folder:")
        self.source_label.pack(pady=(20, 5))

        self.source_entry = ttk.Entry(self, width=50)
        self.source_entry.pack(pady=5)
        self.source_button = ttk.Button(self, text="Browse", command=self.browse_source)
        self.source_button.pack(pady=5)

        self.target_label = ttk.Label(self, text="Target Folder:")
        self.target_label.pack(pady=5)

        self.target_entry = ttk.Entry(self, width=50)
        self.target_entry.pack(pady=5)
        self.target_button = ttk.Button(self, text="Browse", command=self.browse_target)
        self.target_button.pack(pady=5)

        self.process_button = ttk.Button(self, text="Process Folder", command=self.start_processing)
        self.process_button.pack(pady=(10, 20))

        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(pady=5, fill=tk.X, expand=True)

    def browse_source(self):
        folder_selected = filedialog.askdirectory()
        self.source_entry.delete(0, tk.END)
        self.source_entry.insert(0, folder_selected)

    def browse_target(self):
        folder_selected = filedialog.askdirectory()
        self.target_entry.delete(0, tk.END)
        self.target_entry.insert(0, folder_selected)

    
    def update_progress(self, current, total):
        progress = int((current / total) * 100)
        self.progress['value'] = progress
        self.update_idletasks()

    def completion_message(self):
        messagebox.showinfo("Process Complete", "All files have been processed.")

    def start_processing(self):
        source_folder = self.source_entry.get()
        target_folder = self.target_entry.get()
        if not source_folder or not target_folder:
            messagebox.showerror("Error", "Please specify both source and target folders.")
            return  # This return will exit the method if either folder is not specified.

    # Move the threading call inside the else block or remove the return
        threading.Thread(target=self.process_folder_thread, args=(source_folder, target_folder, self.stored_encodings, self.update_progress, self.completion_message)).start()

# This will replace the placeholder process_folder_thread function in your app
    def process_folder_thread(self, source_folder, target_folder, stored_encodings, update_progress_callback, completion_callback):
        process_folder(source_folder, target_folder, stored_encodings, update_progress_callback, completion_callback)
   
# Running the app
if __name__ == "__main__":
    app = FaceSorterApp()
    app.mainloop()

