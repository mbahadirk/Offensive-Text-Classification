import customtkinter as ctk
from tkinter import messagebox
import pickle
import preprocess
import zeyrek
from googleapiclient.discovery import build
from get_models import getOptions

import re
import html

# Zeyrek analizörü oluştur
analyzer = zeyrek.MorphAnalyzer()

# YouTube API anahtarı
API_KEY = "AIzaSyDnLcG-NAVZ0vpZo-N49yKy379FWW35bvA"


def clean_html_tags_and_time(text):
    """HTML etiketlerini, saat-dakika formatındaki zaman ifadelerini ve HTML karakter referanslarını temizler."""
    # HTML etiketlerini temizle
    clean_text = re.sub(r'<.*?>', '', text)  # HTML etiketlerini kaldır
    # Saat-dakika formatını temizle
    clean_text = re.sub(r'\b\d{1,2}:\d{2}\b', '', clean_text)
    # HTML karakter referanslarını çöz (&#39; -> ')
    clean_text = html.unescape(clean_text)
    return clean_text


def fetch_comments(video_url, comment_size):
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

        while len(comments) < comment_size:  # Yorum sayısına göre döngü
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=10,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                # HTML etiketlerini, saat-dakika ifadelerini ve HTML karakter referanslarını temizle
                clean_comment = clean_html_tags_and_time(comment)
                # Eğer yorum sadece boşluk veya saat/dakika içeriyorsa, bunu atla
                if clean_comment.strip():  # Boş olmayan yorumları ekle
                    comments.append(clean_comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return comments[:comment_size]  # Belirtilen sayıda yorumu döndür
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
    sentence_result_label.configure(text=f"Tahmin: {label}")


def classify_comments():
    video_url = url_entry.get()

    if not video_url.strip():
        messagebox.showwarning("Uyarı", "YouTube video URL'si girmelisiniz!")
        return

    try:
        comment_size = int(comment_size_entry.get())  # Kullanıcının girdiği yorum sayısını al
    except ValueError:
        messagebox.showwarning("Uyarı", "Lütfen geçerli bir sayı girin!")
        return

    # Yorumları getir
    comments = fetch_comments(video_url, comment_size)

    if not comments:
        return

    selected_option = dropdown_var.get()

    # Model ve vektörleştiriciyi yükle
    with open(f'models/{selected_option}.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open(f'models/{selected_option[:-6]}_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    comment_result_textbox.delete(1.0, ctk.END)  # Eski sonuçları temizle

    for comment in comments:
        processed_comment = preprocess.full_cleaning_pipeline(comment)
        vectorized_comment = vectorizer.transform([processed_comment])
        prediction = model.predict(vectorized_comment)

        # Tahmine göre renk belirle
        label = "Toxic" if prediction[0] == 1 else "Non-Toxic"
        color = "#FF0000" if prediction[0] == 1 else "#008000"  # Toxic için kırmızı, Non-Toxic için yeşil

        # Sonucu renkli olarak ekle
        comment_result_textbox.insert(ctk.END, f"Yorum : {comment}\n", (f"default",))
        comment_result_textbox.insert(ctk.END, f"{label}\n\n", (f"{label}",))
        comment_result_textbox.tag_config(f"{label}", foreground=color)


def classify_comments_all_models():
    video_url = url_entry.get()

    if not video_url.strip():
        messagebox.showwarning("Uyarı", "YouTube video URL'si girmelisiniz!")
        return

    try:
        comment_size = int(comment_size_entry.get())  # Kullanıcının girdiği yorum sayısını al
    except ValueError:
        messagebox.showwarning("Uyarı", "Lütfen geçerli bir sayı girin!")
        return

    # Yorumları getir
    comments = fetch_comments(video_url, comment_size)

    if not comments:
        return

    all_models = getOptions()

    # Tüm modelleri yükle
    models = []
    vectorizers = []
    for option in all_models:
        with open(f'models/{option}.pkl', 'rb') as model_file:
            models.append((option, pickle.load(model_file)))

        with open(f'models/{option[:-6]}_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizers.append(pickle.load(vectorizer_file))

    results = []

    for comment in comments:
        predictions = []
        for model_name, model in zip(models, vectorizers):
            processed_comment = preprocess.full_cleaning_pipeline(comment)
            vectorized_comment = model.transform([processed_comment])
            prediction = model_name[1].predict(vectorized_comment)
            predictions.append((model_name[0], prediction[0]))

        toxic_models = [model_name for model_name, prediction in predictions if prediction == 1]
        non_toxic_models = [model_name for model_name, prediction in predictions if prediction == 0]

        avg_prediction = sum([prediction for _, prediction in predictions]) / len(predictions)
        label = f' %{avg_prediction * 100:.0f} Toxic' if avg_prediction >= 0.5 else f' %{(1 - avg_prediction) * 100:.0f} Non-Toxic'
        color = "#FF0000" if avg_prediction >= 0.5 else "#008000"  # Toxic için kırmızı, Non-Toxic için yeşil

        results.append((comment, label, color, toxic_models, non_toxic_models))

    # Sonuçları göster
    comment_result_textbox.delete(1.0, ctk.END)
    for i, (comment, label, color, toxic_models, non_toxic_models) in enumerate(results):
        comment_result_textbox.insert(ctk.END, f"{i + 1}. {comment}\n", ("default",))
        comment_result_textbox.insert(ctk.END, f"Tahmin: {label}\n", (label,))
        comment_result_textbox.insert(ctk.END, f"Toksik veren modeller: {', '.join(toxic_models)}\n", (label,))
        comment_result_textbox.insert(ctk.END, f"Toksik vermeyen modeller: {', '.join(non_toxic_models)}\n\n", (label,))
        comment_result_textbox.tag_config(label, foreground=color)


ctk.set_appearance_mode("light")  # Her zaman açık tema

# GUI'yi oluştur
root = ctk.CTk()
root.title("Toxicity Classifier")
root.geometry("600x800")  # Pencere boyutunu ayarla
root.configure(bg="#2E2E2E")  # Koyu gri arka plan

# Font ayarlarını merkezi bir şekilde yapalım
font_style = ("San-Francisco", 12)  # Yazı tipini burada ayarlıyoruz

# Cümle girişi için etiket ve giriş kutusu
sentence_label = ctk.CTkLabel(root, text="Bir cümle girin:", font=(font_style, 14), text_color="black")
sentence_label.pack(pady=10)

sentence_entry = ctk.CTkEntry(root, width=400, font=font_style)
sentence_entry.pack(pady=10)

sentence_classify_button = ctk.CTkButton(root, text="Cümleyi Sınıflandır", command=classify_sentence, font=font_style,
                                         fg_color="#6A5ACD", text_color="white")
sentence_classify_button.pack(pady=10)

sentence_result_label = ctk.CTkLabel(root, text="", font=(font_style, 14), text_color="black")
sentence_result_label.pack(pady=10)

# YouTube video URL ve Yorum Sayısı girişi için etiket ve giriş kutuları
input_frame = ctk.CTkFrame(root)
input_frame.pack(pady=10)

# Dropdown menü için seçenekler
options = getOptions()

dropdown_var = ctk.StringVar(root)
dropdown_var.set(options[0])  # Varsayılan seçenek

dropdown_menu = ctk.CTkOptionMenu(root, variable=dropdown_var, values=options)
dropdown_menu.pack(pady=10)

url_label = ctk.CTkLabel(input_frame, text="YouTube video URL'sini girin:", font=(font_style, 14), text_color="black")
url_label.grid(row=0, column=0, padx=10, pady=10)

url_entry = ctk.CTkEntry(input_frame, width=250, font=font_style)
url_entry.grid(row=0, column=1, padx=10, pady=10)

comment_size_label = ctk.CTkLabel(input_frame, text="Sayı:", font=(font_style, 14), text_color="black")
comment_size_label.grid(row=0, column=2, padx=10, pady=10)

comment_size_entry = ctk.CTkEntry(input_frame, width=40, font=font_style)
comment_size_entry.grid(row=0, column=3, padx=10, pady=10)


# Yorumları sınıflandırma butonu
classify_comments_button = ctk.CTkButton(root, text="Yorumları Sınıflandır", command=classify_comments, font=font_style,
                                         fg_color="#6A5ACD", text_color="white")
classify_comments_button.pack(pady=10)

classify_comments_all_models_button = ctk.CTkButton(root, text="Yorumları Tüm Modellerle Sınıflandır", command=classify_comments_all_models, font=font_style,
                                         fg_color="#6A5ACD", text_color="white")
classify_comments_all_models_button.pack(pady=10)



# Sonuçları göstermek için kaydırılabilir metin alanı
comment_result_textbox = ctk.CTkTextbox(root, wrap=ctk.WORD, width=800, height=1000, font=(font_style, 20),
                                        bg_color="#F0F8FF")
comment_result_textbox.pack(pady=10)

# Tkinter GUI'yi başlat
root.mainloop()
