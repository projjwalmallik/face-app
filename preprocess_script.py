import cv2
import os
import face_recognition
import json
import numpy as np  # Corrected numpy import
from scipy.spatial import distance

def save_encodings(encodings):
    data = [{"encoding": encoding.tolist(), "name": name} for encoding, name in encodings]
    with open('encodings.json', 'w') as file:
        json.dump(data, file)

def preprocess_images(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            image = cv2.imread(filepath)
            resized_image = cv2.resize(image, (600, 600))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filepath, gray_image)

def encode_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        return face_encodings[0]
    return None

def process_and_encode(directory):
    preprocess_images(directory)
    encodings = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            encoding = encode_faces(filepath)
            if encoding is not None:
                encodings.append((encoding, os.path.basename(subdir)))
    return encodings

directory = 'photos'  # Ensure this path is correct
encodings = process_and_encode(directory)
save_encodings(encodings)

