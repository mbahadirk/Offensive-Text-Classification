import tkinter as tk
import torch
import numpy as np
from gensim.models import KeyedVectors
from sentence_to_vec import sentence_to_vec
from fasttext_lstm_classifier import FastTextLSTMClassifier

# Model ve vectorizer dosyalarını yükle
fasttext_model = KeyedVectors.load("cc.tr.300.kv")
model = torch.load("fasttext_lstm_model.pt", weights_only= False, map_location=torch.device("cpu"))
model.eval()

def predict_sentence(sentence):
    vec = sentence_to_vec(sentence, fasttext_model)
    vec = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # (1, 300)
    with torch.no_grad():
        output = model(vec)
        prediction = output.argmax(dim=1).item()
    return prediction

def on_predict():
    sentence = entry.get()
    result = predict_sentence(sentence)
    if result == 1:
        result_label.config(text="🔴 Toxic", fg="red")
    else:
        result_label.config(text="🟢 Toxic değil", fg="green")

# Tkinter GUI
root = tk.Tk()
root.title("Toxic Yorum Tespiti")
root.geometry("400x200")

entry = tk.Entry(root, font=("Arial", 14), width=40)
entry.pack(pady=20)

predict_button = tk.Button(root, text="Tahmin Et", command=on_predict, font=("Arial", 12))
predict_button.pack()

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=20)

root.mainloop()
