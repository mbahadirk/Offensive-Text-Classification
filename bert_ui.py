import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from tkinter.scrolledtext import ScrolledText
from preprocess_derin_raw import full_cleaning_pipeline
import re
import html
from googleapiclient.discovery import build

# Cihaz (sabit CPU)
device = torch.device("cpu")
print(f"Cihaz: {device}")

# YouTube API anahtarı
API_KEY = "AIzaSyDnLcG-NAVZ0vpZo-N49yKy379FWW35bvA"

# Global değişkenler
model = None  # Modeli global olarak tutacağız
selected_model_path = None  # Seçilen model dosyasının yolu

# Tokenizer'ı yükle
try:
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
    print("Tokenizer yüklendi.")
except Exception as e:
    print(f"Tokenizer yüklenirken hata: {e}")
    exit()

# Model sınıfı (bert2con.pt ile uyumlu)
class BERTClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)  # 2 sınıf için

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        out = self.dropout(pooled_output)
        out = self.linear(out)
        return out  # Ham logits döndürüyor

# HTML etiketlerini ve zaman formatlarını temizleme
def clean_html_tags_and_time(text):
    """HTML etiketlerini, saat-dakika formatındaki zaman ifadeleri ve HTML karakter referanslarını temizler."""
    # HTML etiketlerini temizle
    clean_text = re.sub(r'<.*?>', '', text)
    # Saat-dakika formatını temizle
    clean_text = re.sub(r'\b\d{1,2}:\d{2}\b', '', clean_text)
    # HTML karakter referanslarını çöz (&#39; -> ')
    clean_text = html.unescape(clean_text)
    return clean_text.strip()

# YouTube yorumlarını çekme
def fetch_comments(video_url, comment_size):
    try:
        # YouTube video ID'yi URL'den ayıkla
        video_id = video_url.split("v=")[1]
        if "&" in video_id:
            video_id = video_id.split("&")[0]

        # YouTube Data API servisini oluştur
        youtube = build("youtube", "v3", developerKey=API_KEY)

        # Yorumları çek
        comments = []
        next_page_token = None

        while len(comments) < comment_size:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,  # API'den bir seferde max 100 yorum
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                clean_comment = clean_html_tags_and_time(comment)
                if clean_comment:  # Boş olmayan yorumları ekle
                    comments.append(clean_comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return comments[:comment_size]  # Belirtilen sayıda yorumu döndür
    except Exception as e:
        messagebox.showerror("Hata", f"Yorumları çekerken hata: {e}")
        print(f"Yorum çekme hatası: {e}")
        return []

# Modeli yükleme fonksiyonu
def load_model():
    global model, selected_model_path
    file_path = filedialog.askopenfilename(
        title="Model Dosyası Seç",
        filetypes=[("PyTorch Model Files", "*.pt")]
    )
    if file_path:
        try:
            model = BERTClassifier()
            model.load_state_dict(torch.load(file_path, map_location=device))
            model.to(device)
            model.eval()
            selected_model_path = file_path
            model_label.config(text=f"Seçilen Model: {file_path.split('/')[-1]}")
            print(f"Model yüklendi: {file_path}")
        except Exception as e:
            model = None
            selected_model_path = None
            model_label.config(text="Seçilen Model: Yok")
            messagebox.showerror("Hata", f"Model yüklenirken hata: {e}")
            print(f"Model yüklenirken hata: {e}")

# Tek cümle sınıflandırma
def classify_text():
    global model
    if model is None:
        messagebox.showwarning("Uyarı", "Lütfen önce bir model seçin.")
        return

    raw_text = entry.get().strip()
    threshold = threshold_var.get() / 100.0  # Kaydırıcıdan eşik değerini al

    if not raw_text:
        messagebox.showwarning("Uyarı", "Lütfen bir cümle girin.")
        return

    try:
        # Metni ön işleme tabi tut
        cleaned = full_cleaning_pipeline(raw_text)
        print(f"İşlenmiş metin: {cleaned}, Tip: {type(cleaned)}")  # Hata ayıklama

        # cleaned'ın string olduğundan emin ol
        if not isinstance(cleaned, str):
            raise ValueError(f"full_cleaning_pipeline string döndürmeli, ama şu tip döndü: {type(cleaned)}")

        # Tokenize et
        tokens = tokenizer(
            cleaned,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Tahmin yap
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)  # Softmax ile olasılıklar
            prob_toxic = probs[0, 1].item()  # Toksik sınıfın olasılığı
            prob_non_toxic = probs[0, 0].item()  # Zararsız sınıfın olasılığı
            prediction = 1 if prob_toxic >= threshold else 0
            label = "Toksik (1)" if prediction == 1 else "Zararsız (0)"

        # Sonucu göster
        result_label.config(
            text=f"Tahmin: {label}\nToksik Olasılık: {prob_toxic:.4f}\nZararsız Olasılık: {prob_non_toxic:.4f}\nİşlenmiş Metin: {cleaned}"
        )

        # Olasılık çubuğunu güncelle
        progress_var.set(prob_toxic * 100)
        progress_label.config(text=f"Toksiklik Olasılığı: %{prob_toxic * 100:.1f}")

        # Tahmini logla
        with open("predictions.log", "a", encoding="utf-8") as f:
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
    global model
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

    threshold = threshold_var.get() / 100.0  # Kaydırıcıdan eşik değerini al

    # Yorumları çek
    comments = fetch_comments(video_url, comment_size)
    if not comments:
        return

    # Sonuçları temizle
    comment_result_text.delete("1.0", tk.END)

    # Yorumları sınıflandır
    for i, comment in enumerate(comments, 1):
        try:
            cleaned = full_cleaning_pipeline(comment)
            if not cleaned:
                continue  # Boşsa atla

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

            # Renk belirle
            color = "red" if prediction == 1 else "green"

            # Sonucu metin kutusuna ekle
            comment_result_text.insert(tk.END, f"Yorum {i}: {comment}\n")
            comment_result_text.insert(tk.END, f"İşlenmiş: {cleaned}\n")
            comment_result_text.insert(tk.END, f"Tahmin: {label}\n", f"tag_{label}")
            comment_result_text.insert(tk.END, f"Toksik Olasılık: {prob_toxic:.4f}\n")
            comment_result_text.insert(tk.END, f"Zararsız Olasılık: {prob_non_toxic:.4f}\n\n")

            # Renk etiketini tanımla
            comment_result_text.tag_configure(f"tag_{label}", foreground=color)

            # Tahmini logla
            with open("predictions.log", "a", encoding="utf-8") as f:
                f.write(
                    f"YouTube Yorumu: {comment}\nİşlenmiş: {cleaned}\nTahmin: {label}\n"
                    f"Toksik Olasılık: {prob_toxic:.4f}\nZararsız Olasılık: {prob_non_toxic:.4f}\n"
                    f"Eşik: {threshold:.2f}\nModel: {selected_model_path}\nVideo URL: {video_url}\n\n"
                )

        except Exception as e:
            comment_result_text.insert(tk.END, f"Yorum {i}: {comment}\n")
            comment_result_text.insert(tk.END, f"Hata: {e}\n\n")
            print(f"Yorum sınıflandırma hatası: {e}")

# Arayüzü temizleme fonksiyonu
def clear_text():
    entry.delete(0, tk.END)
    url_entry.delete(0, tk.END)
    comment_size_entry.delete(0, tk.END)
    comment_size_entry.insert(0, "50")  # Varsayılan 50
    result_label.config(text="")
    comment_result_text.delete("1.0", tk.END)
    progress_var.set(0)
    progress_label.config(text="Toksiklik Olasılığı: %0.0")
    threshold_var.set(50)

# Tkinter arayüzü
root = tk.Tk()
root.title("Türkçe Toksiklik Sınıflayıcı")
root.geometry("600x800")  # Yorumlar için daha büyük alan
root.resizable(False, False)

# Stil
style = ttk.Style()
style.configure("TButton", font=("Arial", 12))
style.configure("TLabel", font=("Arial", 12))

# Çerçeve
frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

# Model seçme
model_frame = ttk.Frame(frame)
model_frame.pack(fill=tk.X, pady=5)
ttk.Button(model_frame, text="Model Seç", command=load_model).pack(side=tk.LEFT, padx=5)
model_label = ttk.Label(model_frame, text="Seçilen Model: Yok")
model_label.pack(side=tk.LEFT, padx=5)

# Cümle girişi
ttk.Label(frame, text="Cümleyi girin:").pack(anchor="w")
entry = ttk.Entry(frame, width=50, font=("Arial", 12))
entry.pack(pady=5, fill=tk.X)

# YouTube URL ve yorum sayısı girişi
url_frame = ttk.Frame(frame)
url_frame.pack(fill=tk.X, pady=5)
ttk.Label(url_frame, text="YouTube Video URL:").pack(anchor="w")
url_entry = ttk.Entry(url_frame, width=50, font=("Arial", 12))
url_entry.pack(pady=5, fill=tk.X)

ttk.Label(url_frame, text="Yorum Sayısı:").pack(anchor="w")
comment_size_entry = ttk.Entry(url_frame, width=10, font=("Arial", 12))
comment_size_entry.pack(pady=5, anchor="w")
comment_size_entry.insert(0, "50")  # Varsayılan 50 yorum

# Eşik ayarı
ttk.Label(frame, text="Toksiklik Eşiği (%):").pack(anchor="w")
threshold_var = tk.DoubleVar(value=50.0)  # Varsayılan eşik: 0.5
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

# Yorum sonuçları için kaydırılabilir metin alanı
comment_result_text = ScrolledText(frame, wrap=tk.WORD, width=60, height=15, font=("Arial", 10))
comment_result_text.pack(pady=10, fill=tk.BOTH, expand=True)

root.mainloop()