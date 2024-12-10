import tkinter as tk
from tkinter import messagebox, scrolledtext
import pickle
import preprocess
import zeyrek
from googleapiclient.discovery import build
import os

# Zeyrek analizörü oluştur
analyzer = zeyrek.MorphAnalyzer()

# YouTube API anahtarı
API_KEY = "AIzaSyDnLcG-NAVZ0vpZo-N49yKy379FWW35bvA"


def getOptions():
    # models klasörünün yolu
    models_folder = "models"
    options = []
    # models klasöründeki dosyaları listele
    if os.path.exists(models_folder):
        for file_name in os.listdir(models_folder):
            if file_name.endswith("model.pkl"):
                options.append(os.path.splitext(file_name)[0])
    else:
        print(f"{models_folder} klasörü bulunamadı.")
    return options

def fetch_comments(video_url):
    try:
        # YouTube video ID'yi URL'den ayıklayın
        video_id = video_url.split("v=")[1]
        if "&" in video_id:
            video_id = video_id.split("&")[0]

        # YouTube Data API servisini oluştur
        youtube = build("youtube", "v3", developerKey=API_KEY)

        # Yorumları çekmek için API çağrısı
        comments = []
        next_page_token = None

        while len(comments) < 20:  # En az 20 yorum almak için döngü
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=10,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return comments[:20]  # İlk 20 yorumu döndür
    except Exception as e:
        messagebox.showerror("Hata", f"Yorumları çekerken bir hata oluştu: {e}")
        return []


def classify_sentence():
    user_sentence = sentence_entry.get()

    if not user_sentence.strip():
        messagebox.showwarning("Uyarı", "Cümle girmelisiniz!")
        return

    selected_option = dropdown_var.get()

    # Model ve vektörleştiriciyi yükle
    with open(f'models/{selected_option}.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open(f'models/{selected_option[:-6]}_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Cümleyi temizleyip sınıflandır
    processed_sentence = preprocess.full_cleaning_pipeline(user_sentence)
    vectorized_sentence = vectorizer.transform([processed_sentence])
    prediction = model.predict(vectorized_sentence)

    label = "Toxic" if prediction[0] == 1 else "Non-Toxic"
    sentence_result_label.config(text=f"Tahmin: {label}")

def classify_comments():
    video_url = url_entry.get()

    if not video_url.strip():
        messagebox.showwarning("Uyarı", "YouTube video URL'si girmelisiniz!")
        return

    # Yorumları getir
    comments = fetch_comments(video_url)

    if not comments:
        return

    selected_option = dropdown_var.get()
    
    # Model ve vektörleştiriciyi yükle
    with open(f'models/{selected_option}.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open(f'models/{selected_option[:-6]}_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    results = []

    for comment in comments:
        processed_comment = preprocess.full_cleaning_pipeline(comment)
        vectorized_comment = vectorizer.transform([processed_comment])
        prediction = model.predict(vectorized_comment)

        label = "Toxic" if prediction[0] == 1 else "Non-Toxic"
        results.append((comment, label))

    # Sonuçları göster
    result_text = "".join([f"{i+1}. {comment}\nTahmin: {label}\n\n" for i, (comment, label) in enumerate(results)])
    comment_result_textbox.delete(1.0, tk.END)
    comment_result_textbox.insert(tk.END, result_text)

# GUI'yi oluştur
root = tk.Tk()
root.title("Toxicity Classifier")

# Cümle girişi için etiket ve giriş kutusu
sentence_label = tk.Label(root, text="Bir cümle girin:")
sentence_label.pack(pady=10)

sentence_entry = tk.Entry(root, width=50)
sentence_entry.pack(pady=10)

sentence_classify_button = tk.Button(root, text="Cümleyi Sınıflandır", command=classify_sentence)
sentence_classify_button.pack(pady=10)

sentence_result_label = tk.Label(root, text="", font=("Helvetica", 12))
sentence_result_label.pack(pady=10)

# YouTube video URL girişi için etiket ve giriş kutusu
url_label = tk.Label(root, text="YouTube video URL'sini girin:")
url_label.pack(pady=10)

url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=10)

# Dropdown menü için seçenekler
options = getOptions()

dropdown_var = tk.StringVar(root)
dropdown_var.set(options[0])  # Varsayılan seçenek

dropdown_menu = tk.OptionMenu(root, dropdown_var, *options)
dropdown_menu.pack(pady=10)

# Yorumları sınıflandırma butonu
classify_comments_button = tk.Button(root, text="Yorumları Sınıflandır", command=classify_comments)
classify_comments_button.pack(pady=10)

# Sonuçları göstermek için kaydırılabilir metin alanı
comment_result_textbox = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Helvetica", 10))
comment_result_textbox.pack(pady=10)

# Tkinter GUI'yi başlat
root.mainloop()
