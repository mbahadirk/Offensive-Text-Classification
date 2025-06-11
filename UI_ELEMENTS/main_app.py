import torch
import tkinter as tk
from tkinter import messagebox
import threading
import asyncio

# Diğer modüllerden importlar
try:
    from ui_module import AppUI
    print("UI Module loaded.")
except ImportError:
    messagebox.showerror("Hata", "ui_module.py dosyası bulunamadı. Uygulama başlatılamıyor.")
    exit()

try:
    # ExtendedBinarySentimentClassifier'ı da import edin
    from model_module import ModelManager, BertClassifier, BertLSTMClassifier, ExtendedBinarySentimentClassifier
    print("Model Module loaded.")
except ImportError as e:
    messagebox.showerror("Hata", f"model_module.py yüklenemedi: {e}\nModel yükleme ve sınıflandırma devre dışı.")
    ModelManager = None
    ExtendedBinarySentimentClassifier = None # Hata durumunda bunu da None yapın
    BertClassifier = None
    BertLSTMClassifier = None

# Ön işleme modülünü import edin
try:
    from preprocessing_module import clean_text, full_cleaning_pipeline
    print("Preprocessing Module loaded.")
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    messagebox.showwarning("Uyarı", f"preprocessing_module.py yüklenemedi: {e}. Metin ön işleme devre dışı.")
    clean_html_tags_and_time = None
    full_cleaning_pipeline = None
    PREPROCESSING_AVAILABLE = False
except Exception as e:
    messagebox.showwarning("Uyarı", f"Metin ön işleme yüklenirken hata: {e}. Ön işleme devre dışı.")
    print(f"Preprocessing pipeline loading error: {e}")
    clean_html_tags_and_time = None
    full_cleaning_pipeline = None
    PREPROCESSING_AVAILABLE = False

# Veri çekme modülünü import edin
try:
    from data_fetcher_module import fetch_comments_from_youtube, scrape_tweets_from_twitter
    print("Data Fetcher Module loaded.")
except ImportError as e:
    messagebox.showerror("Hata", f"data_fetcher_module.py yüklenemedi: {e}\nVeri çekme devre dışı.")
    fetch_comments_from_youtube = None
    scrape_tweets_from_twitter = None

try:
    from googleapiclient.discovery import build
    YOUTUBE_API_AVAILABLE = True
    print("Google API client loaded for YouTube.")
except ImportError:
    messagebox.showwarning("Uyarı", "google-api-python-client kütüphanesi yüklü değil. YouTube scraping devre dışı.")
    YOUTUBE_API_AVAILABLE = False
except Exception as e:
    messagebox.showwarning("Uyarı", f"Google API client yüklenirken hata: {e}. YouTube scraping devre dışı.")
    print(f"Google API client loading error: {e}")
    YOUTUBE_API_AVAILABLE = False

try:
    from x3 import TwitterScraper
    print("TwitterScraper modülü yüklendi.")
    TWITTER_SCRAPER_AVAILABLE = True
except ImportError:
    messagebox.showwarning("Uyarı", "x3.py veya TwitterScraper sınıfı bulunamadı. Twitter scraping devre dışı.")
    TWITTER_SCRAPER_AVAILABLE = False
except Exception as e:
     messagebox.showwarning("Uyarı", f"TwitterScraper yüklenirken hata: {e}. Twitter scraping devre dışı.")
     print(f"TwitterScraper yükleme hatası: {e}")
     TWITTER_SCRAPER_AVAILABLE = False


class ToxicityClassifierApp:
    def __init__(self, root):
        self.root = root
        self.device = torch.device("cpu")
        print(f"Cihaz: {self.device}")

        self.API_KEY = "your api key"

        self.model_manager = ModelManager(self.device) if ModelManager else None
        if not self.model_manager:
            self.tokenizer = None
            self.TOKENIZER_AVAILABLE = False
        else:
            self.tokenizer, self.TOKENIZER_AVAILABLE = self.model_manager.get_current_tokenizer()

        self.classified_comments = []
        self.classified_tweets = []

        self.ui = AppUI(root, TWITTER_SCRAPER_AVAILABLE)

        self._bind_ui_elements()
        self._initial_load()

    def _bind_ui_elements(self):
        if self.model_manager:
            
            self.ui.class_combo.bind("<<ComboboxSelected>>", self.on_class_select)
            self.ui.selected_model_type.trace("w", self.on_class_select)

        self.ui.btn_classify_manual.config(command=self.classify_text)
        self.ui.btn_classify_comments.config(command=self.classify_comments)
        self.ui.btn_list_toxic_comments.config(command=self.list_toxic_comments)
        self.ui.btn_list_non_toxic_comments.config(command=self.list_non_toxic_comments)

        if TWITTER_SCRAPER_AVAILABLE:
            self.ui.btn_classify_tweets.config(command=self.classify_tweets_async)
            self.ui.btn_list_toxic_tweets.config(command=self.list_toxic_tweets)
            self.ui.btn_list_non_toxic_tweets.config(command=self.list_non_toxic_tweets)

        self.ui.btn_clear_text.config(command=self.clear_all_data)

        self.ui.threshold_var.trace("w", lambda *args: self.update_summary_labels())

    def _initial_load(self):
     if self.model_manager:
        selected_model_type = self.ui.selected_model_type.get()
        self.model_manager.load_model(selected_model_type, self.ui.update_selected_model_label)
     else:
        self.ui.update_selected_model_label("Yok", self.ui.selected_model_type.get())
     self.ui.update_summary_labels(0, 0)
     
    def on_class_select(self, *args):
     selected_model_type = self.ui.selected_model_type.get()
     if self.model_manager:
        self.model_manager.load_model(selected_model_type, self.ui.update_selected_model_label)
     else:
        messagebox.showerror("Hata", "Model Yöneticisi mevcut değil.")
 

    



    def update_summary_labels(self):
        toxic_count_comments = sum(1 for c in self.classified_comments if c["prediction"] == 1)
        non_toxic_count_comments = len(self.classified_comments) - toxic_count_comments

        toxic_count_tweets = sum(1 for t in self.classified_tweets if t["prediction"] == 1)
        non_toxic_count_tweets = len(self.classified_tweets) - toxic_count_tweets

        self.ui.update_summary_labels(toxic_count_comments + toxic_count_tweets, non_toxic_count_comments + non_toxic_count_tweets)

    def classify_text(self):
        current_model, _ = self.model_manager.get_current_model() if self.model_manager else (None, None)
        tokenizer, tokenizer_available = self.model_manager.get_current_tokenizer() if self.model_manager else (None, False)

        if current_model is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir model seçin.")
            return
        if not tokenizer_available or tokenizer is None:
             messagebox.showerror("Hata", "Tokenizer yüklenemedi. Cümle sınıflandırılamıyor.")
             return
        if not PREPROCESSING_AVAILABLE:
             messagebox.showwarning("Uyarı", "Metin ön işleme kullanılamıyor. Cümle sınıflandırılamıyor.")
             return

        raw_text = self.ui.manual_entry.get().strip()
        threshold = self.ui.threshold_var.get() / 100.0

        if not raw_text:
            messagebox.showwarning("Uyarı", "Lütfen bir cümle girin.")
            return

        try:
            cleaned = full_cleaning_pipeline(raw_text)
            if not isinstance(cleaned, str):
                raise ValueError(f"full_cleaning_pipeline string döndürmeli, ama şu tip döndü: {type(cleaned)}")
            if not cleaned:
                 self.ui.result_text.insert(tk.END, f"Manuel Cümle: {raw_text}\nİşlenmiş: Boş (Temizleme Sonrası)\n---\n\n")
                 return

            model_type = self.ui.selected_model_type.get()
            if model_type in ["none","Bert", "BertLstm", "ExtendedBertLstm"]: # extended_bert_lstm eklendi
                tokens = tokenizer(
                    cleaned,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                with torch.no_grad():
                    logits = current_model(input_ids, attention_mask)
            else:
                 messagebox.showerror("Hata", f"Seçili model türü ({model_type}) desteklenmiyor veya tahmin logic eksik.")
                 return

            probs = torch.softmax(logits, dim=1)
            prob_non_toxic = probs[0, 0].item()
            prob_toxic = probs[0, 1].item()

            prediction = 1 if prob_toxic >= threshold else 0
            label = "Toksik" if prediction == 1 else "Non-Toksik"

            color = "red" if prediction == 1 else "green"
            self.ui.result_text.insert(tk.END, "-- Manuel Cümle Analizi --\n", "section_header")
            self.ui.result_text.insert(tk.END, f"Orijinal Cümle: {raw_text}\n")
            self.ui.result_text.insert(tk.END, f"İşlenmiş Cümle: {cleaned}\n")
            self.ui.result_text.insert(tk.END, f"Tahmin: {label}\n", f"tag_{label}_manual")
            self.ui.result_text.insert(tk.END, f"Toksik Olasılık: {prob_toxic:.4f}\n")
            self.ui.result_text.insert(tk.END, f"Non-Toksik Olasılık: {prob_non_toxic:.4f}\n")
            self.ui.result_text.insert(tk.END, "---\n\n")
            self.ui.result_text.tag_configure("section_header", font=("Segoe UI", 11, "bold"))
            self.ui.result_text.tag_configure(f"tag_{label}_manual", foreground=color)

            with open("predictions.log", "a", encoding="utf-8") as f:
                f.write(
                    f"Cümle: {raw_text}\nİşlenmiş: {cleaned}\nTahmin: {label}\n"
                    f"Toksik Olasılık: {prob_toxic:.4f}\nNon-Toksik Olasılık: {prob_non_toxic:.4f}\nEşik: {threshold:.2f}\n"
                    f"Model: {self.model_manager.selected_model_path}\nKaynak: Tek Cümle\n\n"
                )

        except Exception as e:
            messagebox.showerror("Hata", f"Cümle sınıflandırma hatası: {e}")
            print(f"Cümle sınıflandırma hata detayları: {e}")

    def classify_comments(self):
        current_model, selected_model_path = self.model_manager.get_current_model() if self.model_manager else (None, None)
        tokenizer, tokenizer_available = self.model_manager.get_current_tokenizer() if self.model_manager else (None, False)

        if current_model is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir model seçin.")
            return
        if not tokenizer_available or tokenizer is None:
             messagebox.showerror("Hata", "Tokenizer yüklenemedi. Yorumlar sınıflandırılamıyor.")
             return
        if not YOUTUBE_API_AVAILABLE:
             messagebox.showwarning("Uyarı", "YouTube API kullanılamıyor. Yorumlar sınıflandırılamıyor.")
             return
        if not PREPROCESSING_AVAILABLE:
            messagebox.showwarning("Uyarı", "Metin ön işleme kullanılamıyor. Cümle sınıflandırılamıyor.")
            return

        if not fetch_comments_from_youtube:
            messagebox.showerror("Hata", "YouTube yorumlarını çekme fonksiyonu yüklenemedi.")
            return

        youtube_url = self.ui.youtube_url_entry.get().strip()
        try:
            comment_size = int(self.ui.comment_size_entry.get())
            if comment_size <= 0:
                raise ValueError("Yorum sayısı pozitif olmalı.")
        except ValueError:
            messagebox.showwarning("Uyarı", "Lütfen geçerli bir yorum sayısı girin (YouTube sekmesinde).")
            return

        threshold = self.ui.threshold_var.get() / 100.0

        async def fetch_and_process_comments():
            comments = await fetch_comments_from_youtube(
                youtube_url, comment_size, self.API_KEY, YOUTUBE_API_AVAILABLE, build, clean_text
            )

            if not comments:
                self.root.after(0, lambda: self.ui.result_text.delete("1.0", tk.END))
                self.root.after(0, lambda: self.ui.result_text.insert(tk.END, "YouTube yorumları çekilemedi veya hiç yorum bulunamadı. Lütfen URL'yi ve API anahtarını kontrol edin.\n"))
                self.classified_comments = []
                self.root.after(0, self.update_summary_labels)
                return

            self.root.after(0, lambda: self.ui.result_text.delete("1.0", tk.END))
            self.classified_comments = []
            self.root.after(0, lambda: self.ui.result_text.insert(tk.END, "YouTube Yorumları Çekiliyor ve Sınıflandırılıyor...\n\n", "info_tag"))
            self.root.after(0, lambda: self.ui.result_text.tag_configure("info_tag", foreground="blue"))

            temp_classified_comments = []
            for i, comment in enumerate(comments, 1):
                try:
                    cleaned = full_cleaning_pipeline(comment)
                    if not cleaned:
                        continue

                    model_type = self.ui.selected_model_type.get()
                    if model_type in ["Bert", "BertLstm", "ExtendedBertLstm"]: # extended_bert_lstm eklendi
                        tokens = tokenizer(
                            cleaned,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=512
                        )
                        input_ids = tokens["input_ids"].to(self.device)
                        attention_mask = tokens["attention_mask"].to(self.device)
                        with torch.no_grad():
                            logits = current_model(input_ids, attention_mask)
                    else:
                        print(f"Warning: Classification logic not fully implemented for model type: {model_type}")
                        continue

                    probs = torch.softmax(logits, dim=1)
                    prob_non_toxic = probs[0, 0].item()
                    prob_toxic = probs[0, 1].item()

                    prediction = 1 if prob_toxic >= threshold else 0
                    label = "Toksik" if prediction == 1 else "Non-Toksik"

                    temp_classified_comments.append({
                        "index": i,
                        "original": comment,
                        "cleaned": cleaned,
                        "prob_toxic": prob_toxic,
                        "prob_non_toxic": prob_non_toxic,
                        "label": label,
                        "prediction": prediction
                    })

                    self.root.after(0, self.update_comment_results_ui_incremental, temp_classified_comments[-1], youtube_url, threshold, selected_model_path)

                except Exception as e:
                    print(f"Yorum sınıflandırma hatası (yorum {i}): {e}")
                    self.root.after(0, lambda i=i, comment=comment, e=e: self.ui.result_text.insert(tk.END, f"Yorum {i}: {comment}\nHata: {e}\n\n", "error_tag"))
                    self.ui.result_text.tag_configure("error_tag", foreground="orange")

            self.root.after(0, lambda: self.ui.result_text.insert(tk.END, "\n-- YouTube Yorum Sınıflandırma Tamamlandı --\n", "completion_tag"))
            self.root.after(0, lambda: self.ui.result_text.tag_configure("completion_tag", foreground="blue", font=("Segoe UI", 10, "bold")))
            self.root.after(0, lambda: self.update_classified_comments_and_summary(temp_classified_comments))

        threading.Thread(target=lambda: asyncio.run(fetch_and_process_comments())).start()

    def update_classified_comments_and_summary(self, temp_list):
        self.classified_comments = temp_list
        self.update_summary_labels()

    def update_comment_results_ui_incremental(self, result, youtube_url, threshold, selected_model_path):
        color = "red" if result['prediction'] == 1 else "green"
        self.ui.result_text.insert(tk.END, f"Yorum {result['index']}: {result['original']}\n")
        self.ui.result_text.insert(tk.END, f"İşlenmiş: {result['cleaned']}\n")
        self.ui.result_text.insert(tk.END, f"Tahmin: {result['label']}\n", f"tag_{result['label']}")
        self.ui.result_text.insert(tk.END, f"Toksik Olasılık: {result['prob_toxic']:.4f}\n")
        self.ui.result_text.insert(tk.END, f"Non-Toksik Olasılık: {result['prob_non_toxic']:.4f}\n\n")
        self.ui.result_text.tag_configure(f"tag_{result['label']}", foreground=color)

        with open("predictions.log", "a", encoding="utf-8") as f:
            f.write(
                f"YouTube Yorumu: {result['original']}\nİşlenmiş: {result['cleaned']}\nTahmin: {result['label']}\n"
                f"Toksik Olasılık: {result['prob_toxic']:.4f}\nNon-Toksik Olasılık: {result['prob_non_toxic']:.4f}\n"
                f"Eşik: {threshold:.2f}\nModel: {selected_model_path}\nVideo URL: {youtube_url}\n\n"
            )

    async def scrape_and_classify_tweets_task(self, twitter_url, threshold):
        current_model, selected_model_path = self.model_manager.get_current_model() if self.model_manager else (None, None)
        tokenizer, tokenizer_available = self.model_manager.get_current_tokenizer() if self.model_manager else (None, False)

        if not tokenizer_available or tokenizer is None:
             self.root.after(0, self.update_tweet_results_ui, {"status": "error", "message": "Tokenizer yüklenemedi. Tweetler sınıflandırılamıyor."})
             return
        if current_model is None:
            self.root.after(0, self.update_tweet_results_ui, {"status": "error", "message": "Lütfen önce bir model seçin."})
            return
        if not PREPROCESSING_AVAILABLE:
             messagebox.showwarning("Uyarı", "Metin ön işleme kullanılamıyor. Cümle sınıflandırılamıyor.")
             return
        if not scrape_tweets_from_twitter:
            self.root.after(0, self.update_tweet_results_ui, {"status": "error", "message": "Twitter tweetlerini çekme fonksiyonu yüklenemedi."})
            return

        tweet_replies = await scrape_tweets_from_twitter(
            twitter_url, TWITTER_SCRAPER_AVAILABLE, TwitterScraper
        )

        if isinstance(tweet_replies, str):
            self.root.after(0, self.update_tweet_results_ui, {"status": "error", "message": tweet_replies})
            return

        if not tweet_replies:
            self.root.after(0, self.update_tweet_results_ui, {"status": "info", "message": "Belirtilen URL'den tweet yanıtı çekilemedi."})
            return

        results = []
        for i, tweet_text in enumerate(tweet_replies, 1):
            try:
                cleaned = full_cleaning_pipeline(tweet_text)
                if not cleaned:
                    continue

                model_type = self.ui.selected_model_type.get()

                if model_type in ["Bert", "BertLstm", "ExtendedBertLstm"]: # extended_bert_lstm eklendi
                     tokens = tokenizer(
                         cleaned,
                         return_tensors="pt",
                         padding="max_length",
                         truncation=True,
                         max_length=512
                     )
                     input_ids = tokens["input_ids"].to(self.device)
                     attention_mask = tokens["attention_mask"].to(self.device)
                     with torch.no_grad():
                         logits = current_model(input_ids, attention_mask)
                else:
                     print(f"Warning: Classification logic not fully implemented for model type: {model_type} in tweet scraper.")
                     continue

                probs = torch.softmax(logits, dim=1)
                prob_non_toxic = probs[0, 0].item()
                prob_toxic = probs[0, 1].item()

                prediction = 1 if prob_toxic >= threshold else 0
                label = "Toksik" if prediction == 1 else "Non-Toksik"

                results.append({
                    "index": i,
                    "original": tweet_text,
                    "cleaned": cleaned,
                    "prob_toxic": prob_toxic,
                    "prob_non_toxic": prob_non_toxic,
                    "label": label,
                    "prediction": prediction
                })

                self.root.after(0, self.update_tweet_results_ui, {"status": "result", "data": results[-1], "url": twitter_url, "threshold": threshold, "model_path": selected_model_path})

            except Exception as e:
                 print(f"Tweet sınıflandırma hatası (tweet {i}): {e}")
                 self.root.after(0, self.update_tweet_results_ui, {"status": "tweet_error", "index": i, "original": tweet_text, "error": str(e)})

        self.root.after(0, self.update_tweet_results_ui, {"status": "completed", "data": results})


    def classify_tweets_async(self):
        twitter_url = self.ui.twitter_url_entry.get().strip()
        threshold = self.ui.threshold_var.get() / 100.0

        if not twitter_url:
            messagebox.showwarning("Uyarı", "Lütfen bir Twitter/X URL'si girin.")
            return

        self.ui.result_text.delete("1.0", tk.END)
        self.classified_tweets = []
        self.ui.result_text.insert(tk.END, f"Tweet Yanıtları Çekiliyor ve Sınıflandırılıyor: {twitter_url}\n\n", "info_tag")
        self.ui.result_text.tag_configure("info_tag", foreground="blue")

        threading.Thread(target=lambda: asyncio.run(self.scrape_and_classify_tweets_task(twitter_url, threshold))).start()


    def update_tweet_results_ui(self, result_info):
        if result_info["status"] == "info":
             self.ui.result_text.insert(tk.END, f"{result_info['message']}\n", "info_tag")
             self.ui.result_text.tag_configure("info_tag", foreground="blue")
        elif result_info["status"] == "error":
             self.ui.result_text.insert(tk.END, f"Hata: {result_info['message']}\n", "error_tag")
             self.ui.result_text.tag_configure("error_tag", foreground="red")
        elif result_info["status"] == "tweet_error":
             self.ui.result_text.insert(tk.END, f"Tweet Yanıtı {result_info['index']}: {result_info['original']}\nHata: {result_info['error']}\n\n", "error_tag")
             self.ui.result_text.tag_configure("error_tag", foreground="orange")
        elif result_info["status"] == "result":
            data = result_info["data"]
            color = "red" if data['prediction'] == 1 else "green"
            self.ui.result_text.insert(tk.END, f"Tweet Yanıtı {data['index']}: {data['original']}\n")
            self.ui.result_text.insert(tk.END, f"İşlenmiş: {data['cleaned']}\n")
            self.ui.result_text.insert(tk.END, f"Tahmin: {data['label']}\n", f"tag_{data['label']}_tweet")
            self.ui.result_text.insert(tk.END, f"Toksik Olasılık: {data['prob_toxic']:.4f}\n")
            self.ui.result_text.insert(tk.END, f"Non-Toksik Olasılık: {data['prob_non_toxic']:.4f}\n\n")
            self.ui.result_text.tag_configure(f"tag_{data['label']}_tweet", foreground=color)

            with open("predictions.log", "a", encoding="utf-8") as f:
                f.write(
                    f"Twitter Tweet Yanıtı: {data['original']}\nİşlenmiş: {data['cleaned']}\nTahmin: {data['label']}\n"
                    f"Toksik Olasılık: {data['prob_toxic']:.4f}\nNon-Toksik Olasılık: {data['prob_non_toxic']:.4f}\n"
                    f"Eşik: {result_info['threshold']:.2f}\nModel: {result_info['model_path']}\nTwitter URL: {result_info['url']}\n\n"
                )
        elif result_info["status"] == "completed":
            self.classified_tweets = result_info["data"]
            self.ui.result_text.insert(tk.END, "\n-- Tweet Yanıtı Sınıflandırma Tamamlandı --\n", "completion_tag")
            self.ui.result_text.tag_configure("completion_tag", foreground="blue", font=("Segoe UI", 10, "bold"))
            self.update_summary_labels()

    def list_toxic_comments(self):
        if not self.classified_comments:
            self.ui.result_text.delete("1.0", tk.END)
            self.ui.result_text.insert(tk.END, "Sınıflandırılmış YouTube yorumu bulunamadı. Lütfen önce sınıflandırma yapın.\n")
            return
        threshold = self.ui.threshold_var.get() / 100.0
        self.ui.result_text.delete("1.0", tk.END)
        toxic_comments = [c for c in self.classified_comments if c["prob_toxic"] >= threshold]
        if not toxic_comments:
            self.ui.result_text.insert(tk.END, f"Eşik %{threshold*100:.1f} üzerinde toksik YouTube yorumu bulunamadı.\n")
            return
        self.ui.result_text.insert(tk.END, f"-- Toksik YouTube Yorumları (Eşik: %{threshold*100:.1f}) --\n\n", "section_header")
        self.ui.result_text.tag_configure("section_header", font=("Segoe UI", 10, "bold"), underline=True)

        for comment in toxic_comments:
            self.ui.result_text.insert(tk.END, f"Yorum {comment['index']}: {comment['original']}\n")
            self.ui.result_text.insert(tk.END, f"İşlenmiş: {comment['cleaned']}\n")
            self.ui.result_text.insert(tk.END, f"Tahmin: {comment['label']}\n", f"tag_{comment['label']}")
            self.ui.result_text.insert(tk.END, f"Toksik Olasılık: {comment['prob_toxic']:.4f}\n")
            self.ui.result_text.insert(tk.END, f"Non-Toksik Olasılık: {comment['prob_non_toxic']:.4f}\n\n")
            self.ui.result_text.tag_configure(f"tag_{comment['label']}", foreground="red")

    def list_non_toxic_comments(self):
        if not self.classified_comments:
            self.ui.result_text.delete("1.0", tk.END)
            self.ui.result_text.insert(tk.END, "Sınıflandırılmış YouTube yorumu bulunamadı. Lütfen önce sınıflandırma yapın.\n")
            return
        threshold = self.ui.threshold_var.get() / 100.0
        self.ui.result_text.delete("1.0", tk.END)
        non_toxic_comments = [c for c in self.classified_comments if c["prob_toxic"] < threshold]
        if not non_toxic_comments:
            self.ui.result_text.insert(tk.END, f"Eşik %{threshold*100:.1f} altında Non-Toksik YouTube yorumu bulunamadı.\n")
            return
        self.ui.result_text.insert(tk.END, f"-- Non-Toksik YouTube Yorumları (Eşik: %{threshold*100:.1f}) --\n\n", "section_header")
        self.ui.result_text.tag_configure("section_header", font=("Segoe UI", 10, "bold"), underline=True)
        for comment in non_toxic_comments:
            self.ui.result_text.insert(tk.END, f"Yorum {comment['index']}: {comment['original']}\n")
            self.ui.result_text.insert(tk.END, f"İşlenmiş: {comment['cleaned']}\n")
            self.ui.result_text.insert(tk.END, f"Tahmin: {comment['label']}\n", f"tag_{comment['label']}")
            self.ui.result_text.insert(tk.END, f"Toksik Olasılık: {comment['prob_toxic']:.4f}\n")
            self.ui.result_text.insert(tk.END, f"Non-Toksik Olasılık: {comment['prob_non_toxic']:.4f}\n\n")
            self.ui.result_text.tag_configure(f"tag_{comment['label']}", foreground="green")

    def list_toxic_tweets(self):
        if not self.classified_tweets:
            self.ui.result_text.delete("1.0", tk.END)
            self.ui.result_text.insert(tk.END, "Sınıflandırılmış Twitter tweet yanıtı bulunamadı. Lütfen önce sınıflandırma yapın.\n")
            return
        threshold = self.ui.threshold_var.get() / 100.0
        self.ui.result_text.delete("1.0", tk.END)
        toxic_tweets = [t for t in self.classified_tweets if t["prob_toxic"] >= threshold]
        if not toxic_tweets:
            self.ui.result_text.insert(tk.END, f"Eşik %{threshold*100:.1f} üzerinde toksik Twitter tweet yanıtı bulunamadı.\n")
            return
        self.ui.result_text.insert(tk.END, f"-- Toksik Twitter Tweet Yanıtları (Eşik: %{threshold*100:.1f}) --\n\n", "section_header")
        self.ui.result_text.tag_configure("section_header", font=("Segoe UI", 10, "bold"), underline=True)
        for tweet in toxic_tweets:
            self.ui.result_text.insert(tk.END, f"Tweet Yanıtı {tweet['index']}: {tweet['original']}\n")
            self.ui.result_text.insert(tk.END, f"İşlenmiş: {tweet['cleaned']}\n")
            self.ui.result_text.insert(tk.END, f"Tahmin: {tweet['label']}\n", f"tag_{tweet['label']}_tweet")
            self.ui.result_text.insert(tk.END, f"Toksik Olasılık: {tweet['prob_toxic']:.4f}\n")
            self.ui.result_text.insert(tk.END, f"Non-Toksik Olasılık: {tweet['prob_non_toxic']:.4f}\n\n")
            self.ui.result_text.tag_configure(f"tag_{tweet['label']}_tweet", foreground="red")

    def list_non_toxic_tweets(self):
        if not self.classified_tweets:
            self.ui.result_text.delete("1.0", tk.END)
            self.ui.result_text.insert(tk.END, "Sınıflandırılmış Twitter tweet yanıtı bulunamadı. Lütfen önce sınıflandırma yapın.\n")
            return
        threshold = self.ui.threshold_var.get() / 100.0
        self.ui.result_text.delete("1.0", tk.END)
        non_toxic_tweets = [t for t in self.classified_tweets if t["prob_toxic"] < threshold]
        if not non_toxic_tweets:
            self.ui.result_text.insert(tk.END, f"Eşik %{threshold*100:.1f} altında Non-Toksik Twitter tweet yanıtı bulunamadı.\n")
            return
        self.ui.result_text.insert(tk.END, f"-- Non-Toksik Twitter Tweet Yanıtları (Eşik: %{threshold*100:.1f}) --\n\n", "section_header")
        self.ui.result_text.tag_configure("section_header", font=("Segoe UI", 10, "bold"), underline=True)
        for tweet in non_toxic_tweets:
            self.ui.result_text.insert(tk.END, f"Tweet Yanıtı {tweet['index']}: {tweet['original']}\n")
            self.ui.result_text.insert(tk.END, f"İşlenmiş: {tweet['cleaned']}\n")
            self.ui.result_text.insert(tk.END, f"Tahmin: {tweet['label']}\n", f"tag_{tweet['label']}_tweet")
            self.ui.result_text.insert(tk.END, f"Toksik Olasılık: {tweet['prob_toxic']:.4f}\n")
            self.ui.result_text.insert(tk.END, f"Non-Toksik Olasılık: {tweet['prob_non_toxic']:.4f}\n\n")
            self.ui.result_text.tag_configure(f"tag_{tweet['label']}_tweet", foreground="green")

    def clear_all_data(self):
        self.ui.clear_all_inputs_and_results()
        self.classified_comments = []
        self.classified_tweets = []
        self.update_summary_labels()

if __name__ == "__main__":
    root = tk.Tk()
    app = ToxicityClassifierApp(root)
    root.mainloop()
