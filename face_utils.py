import face_recognition
import numpy as np
import os
import pickle

def encode_face(image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    return encoding[0] if encoding else None

def save_embedding(name, encoding, path="embeddings/"):
    with open(os.path.join(path, f"{name}.pkl"), "wb") as f:
        pickle.dump(encoding, f)

def load_all_embeddings(path="embeddings/"):
    embeddings = {}
    for file in os.listdir(path):
        if file.endswith(".pkl"):
            with open(os.path.join(path, file), "rb") as f:
                embeddings[file.replace(".pkl", "")] = pickle.load(f)
    return embeddings

def match_face(face_encoding, known_encodings, threshold=0.6):
    for name, encoding in known_encodings.items():
        dist = np.linalg.norm(face_encoding - encoding)
        if dist < threshold:
            return name
    return None
