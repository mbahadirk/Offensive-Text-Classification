import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoModel
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from preprocess_derin_raw import full_cleaning_pipeline
import re
import html
from googleapiclient.discovery import build
import os
import numpy as np
from gensim.models import KeyedVectors
import threading
import time

# Cihaz
device = torch.device("cpu")
print(f"Cihaz: {device}")

# YouTube API anahtarı
API_KEY = "AIzaSyDnLcG-NAVZ0vpZo-N49yKy379FWW35bvA"

# Global değişkenler
model = None
selected_model_path = None
selected_model_type = None
classified_comments = []

# BERT Tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained("../models/embeddings/bert-turkish-tokenizer")
    print("BERT Tokenizer yüklendi.")
except Exception as e:
    print(f"BERT Tokenizer yüklenirken hata: {e}")
    exit()


# BERT Model sınıfı
class BertClassifier(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained("../models/embeddings/bert-turkish-model")
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


class BertLSTMClassifier(torch.nn.Module):
    def __init__(self, dropout=0.7):
        super(BertLSTMClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("../models/embeddings/bert-turkish-model")
        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=256, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(256 * 2, 2)  # 2 çünkü Bidirectional LSTM
        self.relu = torch.nn.ReLU()

    def forward(self, input_ids, attention_mask):
        last_hidden_state, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        lstm_output, _ = self.lstm(last_hidden_state)  # (batch_size, seq_len, 2*hidden_size)
        cls_lstm_output = lstm_output[:, 0, :]  # İlk token'ın ([CLS]) çıktısı
        dropout_output = self.dropout(cls_lstm_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


class FastTextLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, output_dim=2, num_layers=1):
        super(FastTextLSTMClassifier, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # FastText ağırlıkları sabit
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embeds = self.embedding(x)  # (batch, max_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(embeds)
        out = self.fc(h_n[-1])
        return out


# HTML etiketlerini ve zaman formatlarını temizleme
def clean_html_tags_and_time(text):
    clean_text = re.sub(r'<.*?>', '', text)
    clean_text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text)
    clean_text = html.unescape(clean_text)
    return clean_text.strip()


# YouTube yorumlarını çekme
def fetch_comments(video_url, comment_size):
    try:
        video_id = video_url.split("v=")[1]
        if "&" in video_id:
            video_id = video_id.split("&")[0]
        youtube = build("youtube", "v3", developerKey=API_KEY)
        comments = []
        next_page_token = None
        while len(comments) < comment_size:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            )
            response = request.execute()
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                clean_comment = clean_html_tags_and_time(comment)
                if clean_comment:
                    comments.append(clean_comment)
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        return comments[:comment_size]
    except Exception as e:
        messagebox.showerror("Hata", f"Yorumları çekerken hata: {e}")
        print(f"Yorum çekme hatası: {e}")
        return []


# Model dosyalarını listeleme
def get_model_files():
    model_dir = "../models/DNN Models"
    try:
        files = os.listdir(model_dir)
        return [f for f in files if f.endswith('.pt')]
    except FileNotFoundError:
        messagebox.showerror("Hata", f"Model klasörü bulunamadı: {model_dir}")
        return []
    except Exception as e:
        messagebox.showerror("Hata", f"Model dosyaları alınırken hata: {e}")
        return []


# Tkinter arayüzü
root = tk.Tk()
root.title("Türkçe Toksiklik Sınıflayıcı")
root.geometry("600x800")
root.resizable(False, False)

# Stil
style = ttk.Style()
style.configure("TButton", font=("Arial", 12))
style.configure("TLabel", font=("Arial", 12))

# Çerçeve
frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

# Global değişkenler (frame tanımlandıktan sonra)
model = None
selected_model_path = None
selected_model_type = tk.StringVar(value="bert")  # Varsayılan olarak "bert" seçili
classified_comments = []

# Model ve sınıf seçimi için çerçeve
model_frame = ttk.Frame(frame)
model_frame.pack(fill=tk.X, pady=5)


def load_model(file_name):
    global model, selected_model_path, selected_model_type
    if not file_name:
        return
    file_path = os.path.join("../models/DNN Models", file_name)
    try:
        # Seçilen sınıf türüne göre model nesnesini oluştur
        model_type = selected_model_type.get()
        if model_type == "bert":
            model = BertClassifier(dropout=0.5)
        elif model_type == "bert_lstm":
            model = BertLSTMClassifier(dropout=0.7)
        else:
            raise ValueError(f"Geçersiz model türü: {model_type}")

        # Modeli cihaza taşı
        model = model.to(device)

        # state_dict'i yükle
        state_dict = torch.load(file_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        # Modeli değerlendirme moduna al
        model.eval()
        selected_model_path = file_path
        selected_label.config(text=f"Seçilen: {file_name} ({model_type})")
        print(f"Model yüklendi: {file_path} ({model_type})")
    except Exception as e:
        model = None
        selected_model_path = None
        selected_label.config(text=f"Seçilen: Yok ({selected_model_type.get()})")
        messagebox.showerror("Hata", f"Model yüklenirken hata: {e}")
        print(f"Model yüklenirken hata: {e}")


# Sınıf seçimi
class_label = ttk.Label(model_frame, text="Sınıf Seç:")
class_label.pack(side=tk.LEFT, padx=5)
class_menu = ttk.Combobox(model_frame, textvariable=selected_model_type, state="readonly")
class_menu['values'] = ("bert", "bert_lstm", "fasttext")
class_menu.pack(side=tk.LEFT, padx=5)
class_menu.current(0)  # Varsayılan olarak "bert" seçili

# Model seçimi
model_label = ttk.Label(model_frame, text="Model Dosyası Seç:")
model_label.pack(side=tk.LEFT, padx=5)
model_menu = ttk.Combobox(model_frame, state="readonly")
model_menu.pack(side=tk.LEFT, padx=5)

# Model dosyalarını yükle
model_files = get_model_files()
if model_files:
    model_menu['values'] = model_files
    model_menu.current(0)  # Varsayılan olarak ilk model dosyası seçili
else:
    model_menu['values'] = ["Model Bulunamadı"]
    model_menu.current(0)
    model_menu.config(state="disabled")

# Seçilen model ve sınıfı göstermek için etiket
selected_label = ttk.Label(model_frame, text="Seçilen: Yok (bert)")
selected_label.pack(side=tk.LEFT, padx=5)


# Model seçimi değiştiğinde çalışacak fonksiyon
def on_model_select(event):
    file_name = model_menu.get()
    if file_name and file_name != "Model Bulunamadı":
        load_model(file_name)
        selected_label.config(text=f"Seçilen: {file_name} ({selected_model_type.get()})")


# Sınıf seçimi değiştiğinde etiketi güncelle
def on_class_select(*args):
    file_name = model_menu.get()
    if file_name and file_name != "Model Bulunamadı":
        selected_label.config(text=f"Seçilen: {file_name} ({selected_model_type.get()})")
    else:
        selected_label.config(text=f"Seçilen: Yok ({selected_model_type.get()})")


# Combobox değişikliklerini bağla
model_menu.bind("<<ComboboxSelected>>", on_model_select)
selected_model_type.trace("w", on_class_select)


# Tek cümle sınıflandırma
def classify_text():
    global model, selected_model_type
    if model is None:
        messagebox.showwarning("Uyarı", "Lütfen önce bir model seçin.")
        return
    raw_text = entry.get().strip()
    threshold = threshold_var.get() / 100.0
    if not raw_text:
        messagebox.showwarning("Uyarı", "Lütfen bir cümle girin.")
        return
    try:
        cleaned = full_cleaning_pipeline(raw_text)
        if not isinstance(cleaned, str):
            raise ValueError(f"full_cleaning_pipeline string döndürmeli, ama şu tip döndü: {type(cleaned)}")
        model_type = selected_model_type.get()
        if model_type in ["bert", "bert_lstm"]:
            tokens = tokenizer(
                cleaned,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            with torch.no_grad():
                logits = model(input_ids, attention_mask)

        probs = torch.softmax(logits, dim=1)
        prob_toxic = probs[0, 1].item()
        prob_non_toxic = probs[0, 0].item()
        prediction = 1 if prob_toxic >= threshold else 0
        label = "Toksik (1)" if prediction == 1 else "Zararsız (0)"
        result_label.config(
            text=f"Tahmin: {label}\nToksik Olasılık: {prob_toxic:.4f}\nZararsız Olasılık: {prob_non_toxic:.4f}\nİşlenmiş Metin: {cleaned}"
        )
        progress_var.set(prob_toxic * 100)
        progress_label.config(text=f"Toksiklik Olasılığı: %{prob_toxic * 100:.1f}")
        with open("../predictions.log", "a", encoding="utf-8") as f:
            f.write(
                f"Cümle: {raw_text}\nİşlenmiş: {cleaned}\nTahmin: {label}\n"
                f"Toksik Olasılık: {prob_toxic:.4f}\nZararsız Olasılık: {prob_non_toxic:.4f}\nEşik: {threshold:.2f}\n"
                f"Model: {selected_model_path}\n\n"
            )
    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
        print(f"Hata detayları: {e}")


# YouTube yorumlarını sınıflandırma
def classify_comments():
    global model, selected_model_type, classified_comments
    if model is None:
        messagebox.showwarning("Uyarı", "Lütfen önce bir model seçin.")
        return
    video_url = url_entry.get().strip()
    if not video_url:
        messagebox.showwarning("Uyarı", "Lütfen bir YouTube video URL'si girin.")
        return
    try:
        comment_size = int(comment_size_entry.get())
        if comment_size <= 0:
            raise ValueError("Yorum sayısı pozitif olmalı.")
    except ValueError:
        messagebox.showwarning("Uyarı", "Lütfen geçerli bir yorum sayısı girin.")
        return
    threshold = threshold_var.get() / 100.0
    comments = fetch_comments(video_url, comment_size)
    if not comments:
        return
    comment_result_text.delete("1.0", tk.END)
    classified_comments = []
    for i, comment in enumerate(comments, 1):
        try:
            cleaned = full_cleaning_pipeline(comment)
            if not cleaned:
                continue
            model_type = selected_model_type.get()
            if model_type in ["bert", "bert_lstm"]:
                tokens = tokenizer(
                    cleaned,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens["attention_mask"].to(device)
                with torch.no_grad():
                    logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            prob_toxic = probs[0, 1].item()
            prob_non_toxic = probs[0, 0].item()
            prediction = 1 if prob_toxic >= threshold else 0
            label = "Toksik" if prediction == 1 else "Zararsız"
            classified_comments.append({
                "index": i,
                "original": comment,
                "cleaned": cleaned,
                "prob_toxic": prob_toxic,
                "prob_non_toxic": prob_non_toxic,
                "label": label,
                "prediction": prediction
            })
            color = "red" if prediction == 1 else "green"
            comment_result_text.insert(tk.END, f"Yorum {i}: {comment}\n")
            comment_result_text.insert(tk.END, f"İşlenmiş: {cleaned}\n")
            comment_result_text.insert(tk.END, f"Tahmin: {label}\n", f"tag_{label}")
            comment_result_text.insert(tk.END, f"Toksik Olasılık: {prob_toxic:.4f}\n")
            comment_result_text.insert(tk.END, f"Zararsız Olasılık: {prob_non_toxic:.4f}\n\n")
            comment_result_text.tag_configure(f"tag_{label}", foreground=color)
            with open("../predictions.log", "a", encoding="utf-8") as f:
                f.write(
                    f"YouTube Yorumu: {comment}\nİşlenmiş: {cleaned}\nTahmin: {label}\n"
                    f"Toksik Olasılık: {prob_toxic:.4f}\nZararsız Olasılık: {prob_non_toxic:.4f}\n"
                    f"Eşik: {threshold:.2f}\nModel: {selected_model_path}\nVideo URL: {video_url}\n\n"
                )
        except Exception as e:
            comment_result_text.insert(tk.END, f"Yorum {i}: {comment}\n")
            comment_result_text.insert(tk.END, f"Hata: {e}\n\n")
            print(f"Yorum sınıflandırma hatası: {e}")


# Toksik yorumları listeleme
def list_toxic_comments():
    global classified_comments
    if not classified_comments:
        messagebox.showwarning("Uyarı", "Önce yorumları sınıflandırın.")
        return
    threshold = threshold_var.get() / 100.0
    comment_result_text.delete("1.0", tk.END)
    toxic_comments = [c for c in classified_comments if c["prob_toxic"] >= threshold and c["prediction"] == 1]
    if not toxic_comments:
        comment_result_text.insert(tk.END, f"Eşik %{threshold * 100:.1f} üzerinde toksik yorum bulunamadı.\n")
        return
    for comment in toxic_comments:
        comment_result_text.insert(tk.END, f"Yorum {comment['index']}: {comment['original']}\n")
        comment_result_text.insert(tk.END, f"İşlenmiş: {comment['cleaned']}\n")
        comment_result_text.insert(tk.END, f"Tahmin: {comment['label']}\n", f"tag_{comment['label']}")
        comment_result_text.insert(tk.END, f"Toksik Olasılık: {comment['prob_toxic']:.4f}\n")
        comment_result_text.insert(tk.END, f"Zararsız Olasılık: {comment['prob_non_toxic']:.4f}\n\n")
        comment_result_text.tag_configure(f"tag_{comment['label']}", foreground="red")
    with open("../predictions.log", "a", encoding="utf-8") as f:
        f.write(f"Toksik Yorumlar Listelendi (Eşik: %{threshold * 100:.1f})\n")
        for comment in toxic_comments:
            f.write(
                f"Yorum {comment['index']}: {comment['original']}\n"
                f"İşlenmiş: {comment['cleaned']}\n"
                f"Tahmin: {comment['label']}\n"
                f"Toksik Olasılık: {comment['prob_toxic']:.4f}\n"
                f"Zararsız Olasılık: {comment['prob_non_toxic']:.4f}\n\n"
            )
        f.write("\n")


# Zararsız yorumları listeleme
def list_non_toxic_comments():
    global classified_comments
    if not classified_comments:
        messagebox.showwarning("Uyarı", "Önce yorumları sınıflandırın.")
        return
    threshold = threshold_var.get() / 100.0
    comment_result_text.delete("1.0", tk.END)
    non_toxic_comments = [c for c in classified_comments if c["prob_non_toxic"] >= threshold and c["prediction"] == 0]
    if not non_toxic_comments:
        comment_result_text.insert(tk.END, f"Eşik %{threshold * 100:.1f} üzerinde zararsız yorum bulunamadı.\n")
        return
    for comment in non_toxic_comments:
        comment_result_text.insert(tk.END, f"Yorum {comment['index']}: {comment['original']}\n")
        comment_result_text.insert(tk.END, f"İşlenmiş: {comment['cleaned']}\n")
        comment_result_text.insert(tk.END, f"Tahmin: {comment['label']}\n", f"tag_{comment['label']}")
        comment_result_text.insert(tk.END, f"Toksik Olasılık: {comment['prob_toxic']:.4f}\n")
        comment_result_text.insert(tk.END, f"Zararsız Olasılık: {comment['prob_non_toxic']:.4f}\n\n")
        comment_result_text.tag_configure(f"tag_{comment['label']}", foreground="green")
    with open("../predictions.log", "a", encoding="utf-8") as f:
        f.write(f"Zararsız Yorumlar Listelendi (Eşik: %{threshold * 100:.1f})\n")
        for comment in non_toxic_comments:
            f.write(
                f"Yorum {comment['index']}: {comment['original']}\n"
                f"İşlenmiş: {comment['cleaned']}\n"
                f"Tahmin: {comment['label']}\n"
                f"Toksik Olasılık: {comment['prob_toxic']:.4f}\n"
                f"Zararsız Olasılık: {comment['prob_non_toxic']:.4f}\n\n"
            )
        f.write("\n")


# Arayüzü temizleme
def clear_text():
    entry.delete(0, tk.END)
    url_entry.delete(0, tk.END)
    comment_size_entry.delete(0, tk.END)
    comment_size_entry.insert(0, "50")
    result_label.config(text="")
    comment_result_text.delete("1.0", tk.END)
    progress_var.set(0)
    progress_label.config(text="Toksiklik Olasılığı: %0.0")
    threshold_var.set(50)


# Cümle girişi
ttk.Label(frame, text="Cümleyi girin:").pack(anchor="w")
entry = ttk.Entry(frame, width=50, font=("Arial", 12))
entry.pack(pady=5, fill=tk.X)

# YouTube URL ve yorum sayısı
url_frame = ttk.Frame(frame)
url_frame.pack(fill=tk.X, pady=5)
ttk.Label(url_frame, text="YouTube Video URL:").pack(anchor="w")
url_entry = ttk.Entry(url_frame, width=50, font=("Arial", 12))
url_entry.pack(pady=5, fill=tk.X)
ttk.Label(url_frame, text="Yorum Sayısı:").pack(anchor="w")
comment_size_entry = ttk.Entry(url_frame, width=10, font=("Arial", 12))
comment_size_entry.pack(pady=5, anchor="w")
comment_size_entry.insert(0, "50")

# Eşik ayarı
ttk.Label(frame, text="Toksiklik Eşiği (%):").pack(anchor="w")
threshold_var = tk.DoubleVar(value=50.0)
threshold_slider = ttk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=threshold_var)
threshold_slider.pack(pady=5, fill=tk.X)
threshold_label = ttk.Label(frame, text=f"Eşik: %{threshold_var.get():.1f}")
threshold_label.pack()
threshold_var.trace("w", lambda *args: threshold_label.config(text=f"Eşik: %{threshold_var.get():.1f}"))

# Butonlar
button_frame = ttk.Frame(frame)
button_frame.pack(pady=10)
ttk.Button(button_frame, text="Cümleyi Sınıflandır", command=classify_text).pack(side=tk.LEFT, padx=5)
ttk.Button(button_frame, text="Yorumları Sınıflandır", command=classify_comments).pack(side=tk.LEFT, padx=5)
ttk.Button(button_frame, text="Toksik Yorumları Listele", command=list_toxic_comments).pack(side=tk.LEFT, padx=5)
ttk.Button(button_frame, text="Toksik Olmayan Yorumları Listele", command=list_non_toxic_comments).pack(side=tk.LEFT,
                                                                                                        padx=5)
ttk.Button(button_frame, text="Temizle", command=clear_text).pack(side=tk.LEFT, padx=5)

# Sonuç etiketi (tek cümle için)
result_label = ttk.Label(frame, text="", font=("Arial", 12), wraplength=550)
result_label.pack(pady=10)

# Olasılık çubuğu
progress_var = tk.DoubleVar()
progress = ttk.Progressbar(frame, variable=progress_var, maximum=100, length=500)
progress.pack(pady=5)
progress_label = ttk.Label(frame, text="Toksiklik Olasılığı: %0.0")
progress_label.pack()

# Yorum sonuçları
comment_result_text = ScrolledText(frame, wrap=tk.WORD, width=60, height=15, font=("Arial", 10))
comment_result_text.pack(pady=10, fill=tk.BOTH, expand=True)

root.mainloop()
