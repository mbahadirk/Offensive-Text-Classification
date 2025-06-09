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
import torch.serialization

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

# Sınıf etiketleri
TOPICS = ["siyaset", "dunya", "ekonomi", "kultur", "saglik", "spor", "teknoloji"]

# BERT Tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained("../models/embeddings/bert-turkish-tokenizer")
    print("BERT Tokenizer yüklendi.")
except Exception as e:
    print(f"BERT Tokenizer yüklenirken hata: {e}")
    exit()

# BERT Model sınıfı
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("../models/embeddings/bert-turkish-model")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 7)  # 7 topic için
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

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
    model_dir = "../models/MultiClass"
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
root.title("Türkçe Konu Sınıflayıcı")
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
selected_model_type = tk.StringVar(value="bert")
classified_comments = []

# Model ve sınıf seçimi için çerçeve
model_frame = ttk.Frame(frame)
model_frame.pack(fill=tk.X, pady=5)

def load_model(file_name):
    global model, selected_model_path, selected_model_type
    if not file_name:
        return
    file_path = os.path.join("../models/MultiClass", file_name)
    try:
        # Allowlist custom and transformers model classes
        from transformers.models.bert.modeling_bert import BertModel
        torch.serialization.add_safe_globals([BertClassifier, BertModel])

        # Seçilen sınıf türüne göre model nesnesini oluştur
        model_type = selected_model_type.get()
        if model_type == "bert":
            # Önce weights_only=True ile state_dict yüklemeyi dene
            try:
                state_dict = torch.load(file_path, map_location=device, weights_only=True)
                if isinstance(state_dict, dict):
                    model = BertClassifier(dropout=0.5)
                    model.load_state_dict(state_dict)
                else:
                    raise ValueError(f"Expected state_dict, got {type(state_dict)}")
            except Exception as e:
                # Eğer weights_only=True başarısız olursa, tam modeli yükle
                print(f"weights_only=True başarısız, tam modeli yüklüyorum: {e}")
                loaded_object = torch.load(file_path, map_location=device, weights_only=False)
                if isinstance(loaded_object, BertClassifier):
                    model = loaded_object
                else:
                    raise ValueError(f"Expected BertClassifier or state_dict, got {type(loaded_object)}")
        else:
            raise ValueError(f"Geçersiz model türü: {model_type}")

        # Modeli cihaza taşı
        model = model.to(device)

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



# Model seçimi
model_label = ttk.Label(model_frame, text="Model Dosyası Seç:")
model_label.pack(side=tk.LEFT, padx=5)
model_menu = ttk.Combobox(model_frame, state="readonly")
model_menu.pack(side=tk.LEFT, padx=5)

# Model dosyalarını yükle
model_files = get_model_files()
if model_files:
    model_menu['values'] = model_files
    model_menu.current(0)
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
        if model_type == "bert":
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
            probs = torch.softmax(logits, dim=1)[0]
            max_prob, predicted_idx = torch.max(probs, dim=0)
            predicted_topic = TOPICS[predicted_idx.item()]
            if max_prob.item() < threshold:
                result_label.config(text=f"Tahmin: Güvenilir tahmin yok (Olasılık: {max_prob.item():.4f} < Eşik: {threshold:.2f})\nİşlenmiş Metin: {cleaned}")
                progress_var.set(0)
                progress_label.config(text="Konu Olasılığı: %0.0")
            else:
                prob_text = "\n".join([f"{topic}: {prob.item():.4f}" for topic, prob in zip(TOPICS, probs)])
                result_label.config(
                    text=f"Tahmin: {predicted_topic}\nOlasılık: {max_prob.item():.4f}\nTüm Olasılıklar:\n{prob_text}\nİşlenmiş Metin: {cleaned}"
                )
                progress_var.set(max_prob.item() * 100)
                progress_label.config(text=f"Konu Olasılığı: %{max_prob.item() * 100:.1f}")
            with open("../predictions.log", "a", encoding="utf-8") as f:
                f.write(
                    f"Cümle: {raw_text}\nİşlenmiş: {cleaned}\nTahmin: {predicted_topic}\n"
                    f"Olasılık: {max_prob.item():.4f}\nTüm Olasılıklar: {', '.join([f'{topic}: {prob.item():.4f}' for topic, prob in zip(TOPICS, probs)])}\n"
                    f"Eşik: {threshold:.2f}\nModel: {selected_model_path}\n\n"
                )
        else:
            raise ValueError(f"Geçersiz model türü: {model_type}")
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
            if model_type == "bert":
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
                probs = torch.softmax(logits, dim=1)[0]
                max_prob, predicted_idx = torch.max(probs, dim=0)
                predicted_topic = TOPICS[predicted_idx.item()]
                if max_prob.item() < threshold:
                    continue  # Eşik altında tahminleri atla
                classified_comments.append({
                    "index": i,
                    "original": comment,
                    "cleaned": cleaned,
                    "probs": probs.tolist(),
                    "predicted_topic": predicted_topic,
                    "max_prob": max_prob.item()
                })
                prob_text = ", ".join([f"{topic}: {prob:.4f}" for topic, prob in zip(TOPICS, probs)])
                comment_result_text.insert(tk.END, f"Yorum {i}: {comment}\n")
                comment_result_text.insert(tk.END, f"İşlenmiş: {cleaned}\n")
                comment_result_text.insert(tk.END, f"Tahmin: {predicted_topic}\n", f"tag_{predicted_topic}")
                comment_result_text.insert(tk.END, f"Olasılık: {max_prob.item():.4f}\n")
                comment_result_text.insert(tk.END, f"Tüm Olasılıklar: {prob_text}\n\n")
                comment_result_text.tag_configure(f"tag_{predicted_topic}", foreground="blue")
                with open("../predictions.log", "a", encoding="utf-8") as f:
                    f.write(
                        f"YouTube Yorumu: {comment}\nİşlenmiş: {cleaned}\nTahmin: {predicted_topic}\n"
                        f"Olasılık: {max_prob.item():.4f}\nTüm Olasılıklar: {prob_text}\n"
                        f"Eşik: {threshold:.2f}\nModel: {selected_model_path}\nVideo URL: {video_url}\n\n"
                    )
            else:
                raise ValueError(f"Geçersiz model türü: {model_type}")
        except Exception as e:
            comment_result_text.insert(tk.END, f"Yorum {i}: {comment}\n")
            comment_result_text.insert(tk.END, f"Hata: {e}\n\n")
            print(f"Yorum sınıflandırma hatası: {e}")

# Belirli bir konuya ait yorumları listeleme
def list_topic_comments():
    global classified_comments
    if not classified_comments:
        messagebox.showwarning("Uyarı", "Önce yorumları sınıflandırın.")
        return
    selected_topic = topic_filter_var.get()
    if not selected_topic:
        messagebox.showwarning("Uyarı", "Lütfen bir konu seçin.")
        return
    threshold = threshold_var.get() / 100.0
    comment_result_text.delete("1.0", tk.END)
    topic_comments = [c for c in classified_comments if c["predicted_topic"] == selected_topic and c["max_prob"] >= threshold]
    if not topic_comments:
        comment_result_text.insert(tk.END, f"Konu '{selected_topic}' için eşik %{threshold*100:.1f} üzerinde yorum bulunamadı.\n")
        return
    for comment in topic_comments:
        prob_text = ", ".join([f"{topic}: {prob:.4f}" for topic, prob in zip(TOPICS, comment["probs"])])
        comment_result_text.insert(tk.END, f"Yorum {comment['index']}: {comment['original']}\n")
        comment_result_text.insert(tk.END, f"İşlenmiş: {comment['cleaned']}\n")
        comment_result_text.insert(tk.END, f"Tahmin: {comment['predicted_topic']}\n", f"tag_{comment['predicted_topic']}")
        comment_result_text.insert(tk.END, f"Olasılık: {comment['max_prob']:.4f}\n")
        comment_result_text.insert(tk.END, f"Tüm Olasılıklar: {prob_text}\n\n")
        comment_result_text.tag_configure(f"tag_{comment['predicted_topic']}", foreground="blue")
    with open("../predictions.log", "a", encoding="utf-8") as f:
        f.write(f"Konu '{selected_topic}' Yorumları Listelendi (Eşik: %{threshold*100:.1f})\n")
        for comment in topic_comments:
            prob_text = ", ".join([f"{topic}: {prob:.4f}" for topic, prob in zip(TOPICS, comment["probs"])])
            f.write(
                f"Yorum {comment['index']}: {comment['original']}\n"
                f"İşlenmiş: {comment['cleaned']}\n"
                f"Tahmin: {comment['predicted_topic']}\n"
                f"Olasılık: {comment['max_prob']:.4f}\n"
                f"Tüm Olasılıklar: {prob_text}\n\n"
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
    progress_label.config(text="Konu Olasılığı: %0.0")
    threshold_var.set(50)
    topic_filter_var.set("")

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
ttk.Label(frame, text="Konu Tahmin Eşiği (%):").pack(anchor="w")
threshold_var = tk.DoubleVar(value=50.0)
threshold_slider = ttk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=threshold_var)
threshold_slider.pack(pady=5, fill=tk.X)
threshold_label = ttk.Label(frame, text=f"Eşik: %{threshold_var.get():.1f}")
threshold_label.pack()
threshold_var.trace("w", lambda *args: threshold_label.config(text=f"Eşik: %{threshold_var.get():.1f}"))

# Konu filtresi
ttk.Label(frame, text="Konu Filtresi:").pack(anchor="w")
topic_filter_var = tk.StringVar()
topic_filter_menu = ttk.Combobox(frame, textvariable=topic_filter_var, state="readonly")
topic_filter_menu['values'] = TOPICS
topic_filter_menu.pack(pady=5, fill=tk.X)

# Butonlar
button_frame = ttk.Frame(frame)
button_frame.pack(pady=10)
ttk.Button(button_frame, text="Cümleyi Sınıflandır", command=classify_text).pack(side=tk.LEFT, padx=5)
ttk.Button(button_frame, text="Yorumları Sınıflandır", command=classify_comments).pack(side=tk.LEFT, padx=5)
ttk.Button(button_frame, text="Seçilen Konu Yorumlarını Listele", command=list_topic_comments).pack(side=tk.LEFT, padx=5)
ttk.Button(button_frame, text="Temizle", command=clear_text).pack(side=tk.LEFT, padx=5)

# Sonuç etiketi (tek cümle için)
result_label = ttk.Label(frame, text="", font=("Arial", 12), wraplength=550)
result_label.pack(pady=10)

# Olasılık çubuğu
progress_var = tk.DoubleVar()
progress = ttk.Progressbar(frame, variable=progress_var, maximum=100, length=500)
progress.pack(pady=5)
progress_label = ttk.Label(frame, text="Konu Olasılığı: %0.0")
progress_label.pack()

# Yorum sonuçları
comment_result_text = ScrolledText(frame, wrap=tk.WORD, width=60, height=15, font=("Arial", 10))
comment_result_text.pack(pady=10, fill=tk.BOTH, expand=True)

root.mainloop()