import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

class AppUI:
    def __init__(self, root, twitter_scraper_available=False):
        self.root = root
        self.root.title("Offensive Text Classifier")
        self.root.geometry("900x700")
        self.root.configure(bg="#f9f9f9")

        # Initialize Tkinter variables that are directly tied to UI widgets
        self.selected_model_type = tk.StringVar(value="Bert")
        self.threshold_var = tk.DoubleVar(value=50.0)

        self._setup_styles()
        self._create_widgets(twitter_scraper_available)
    def update_threshold_labels(self):
            value = self.threshold_var.get()
            self.threshold_label.config(text=f"Toksiklik Eşiği (%{value:.0f}):")

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TButton", padding=6, relief="flat", background="#007AFF", foreground="white")
        style.map("TButton", background=[("active", "#005BB5")])
        style.configure("TNotebook.Tab", padding=[10, 5], font=("Segoe UI", 10))
        style.configure("TLabelFrame", background="#ffffff")
        style.configure("TLabel", background="#f9f9f9")
        style.configure("TFrame", background="#f9f9f9")
        style.configure("White.TLabel", background="#ffffff")
        style.configure("White.TFrame", background="#ffffff")
        style.configure("White.TLabelframe", background="#ffffff")
    
    
    def _create_widgets(self, twitter_scraper_available):
        # Frame 1: Tabs, Model Selection, Summary
        frame1 = tk.Frame(self.root, height=300, bg="#f9f9f9")
        frame1.pack(fill="both", expand=False, padx=10, pady=10)

        # Left: Tabbed structure
        tab_parent = ttk.Notebook(frame1)
        tab_parent.pack(side="left", fill="y", padx=(0, 20), pady=5)

        # YouTube Tab
        self.youtube_tab = ttk.Frame(tab_parent)
        tab_parent.add(self.youtube_tab, text="YouTube")
        ttk.Label(self.youtube_tab, text="Video URL girin:", style="White.TLabel").pack(anchor="center", padx=50, pady=20)
        self.youtube_url_entry = ttk.Entry(self.youtube_tab)
        self.youtube_url_entry.pack(fill="x", padx=5)

        self.yt_controls_frame = ttk.Frame(self.youtube_tab, style="White.TFrame")
        self.yt_controls_frame.pack(pady=75, fill="x", padx=5)

        ttk.Label(self.yt_controls_frame, text="Yorum Sayısı:", style="White.TLabel").pack(side="left", padx=2, anchor="w")
        self.comment_size_entry = ttk.Entry(self.yt_controls_frame, width=5)
        self.comment_size_entry.pack(side="left", padx=5, anchor="w")
        self.comment_size_entry.insert(0, "50")

        self.yt_btn_frame = ttk.Frame(self.yt_controls_frame, style="White.TFrame")
        self.yt_btn_frame.pack(side="right", fill="x", expand=True)

        self.btn_classify_comments = ttk.Button(self.yt_btn_frame, text="Hepsini Sınıflandır")
        self.btn_classify_comments.pack(side="left", padx=10)
        self.btn_list_toxic_comments = ttk.Button(self.yt_btn_frame, text="Sadece Toksikler")
        self.btn_list_toxic_comments.pack(side="left", padx=10)
        self.btn_list_non_toxic_comments = ttk.Button(self.yt_btn_frame, text="Sadece Non-Toksikler")
        self.btn_list_non_toxic_comments.pack(side="left", padx=10)

        # Twitter Tab
        self.twitter_tab = ttk.Frame(tab_parent)
        tab_parent.add(self.twitter_tab, text="X")
        ttk.Label(self.twitter_tab, text="Tweet URL girin:", style="White.TLabel").pack(anchor="center", padx=5, pady=30)
        self.twitter_url_entry = ttk.Entry(self.twitter_tab)
        self.twitter_url_entry.pack(fill="x", padx=5)

        self.tw_btn_frame = ttk.Frame(self.twitter_tab, style="White.TFrame")
        self.tw_btn_frame.pack(pady=75)

        if twitter_scraper_available:
            self.btn_classify_tweets = ttk.Button(self.tw_btn_frame, text="Hepsini Sınıflandır")
            self.btn_classify_tweets.pack(side="left", padx=2)
            self.btn_list_toxic_tweets = ttk.Button(self.tw_btn_frame, text="Sadece Toksikler")
            self.btn_list_toxic_tweets.pack(side="left", padx=2)
            self.btn_list_non_toxic_tweets = ttk.Button(self.tw_btn_frame, text="Sadece Non-Toksikler")
            self.btn_list_non_toxic_tweets.pack(side="left", padx=2)
        else:
            ttk.Label(self.tw_btn_frame, text="Twitter Scraping Devre Dışı", foreground="orange").pack()

        # Manual Tab
        self.manual_tab = ttk.Frame(tab_parent)
        tab_parent.add(self.manual_tab, text="Manuel")
        ttk.Label(self.manual_tab, text="Cümle giriniz:", style="White.TLabel").pack(anchor="center", padx=30, pady=30)
        self.manual_entry = tk.Entry(self.manual_tab, font=("Segoe UI", 10))
        self.manual_entry.pack(fill="x", padx=5)
        self.btn_classify_manual = ttk.Button(self.manual_tab, text="Analiz Et")
        self.btn_classify_manual.pack(pady=30   )

        # Right: Model and class selection, summary, export
        right_panel = ttk.Frame(frame1, style="White.TFrame")
        right_panel.pack(side="left", fill="both", expand=True, padx=10)

        ttk.Label(right_panel, text="Model Sınıfı Seç:", font=("Segoe UI", 11, "bold"), style="White.TLabel").pack(anchor="w", padx=5, pady=(5, 0))
        self.class_combo = ttk.Combobox(right_panel, values=["Bert", "BertLstm","ExtendedBertLstm"], state="readonly", textvariable=self.selected_model_type)
        self.class_combo.set("Bert")
        self.class_combo.pack(fill="x", padx=5, pady=5)

        

        self.selected_label = ttk.Label(right_panel, text="Seçilen: Yok (Bert)", style="White.TLabel")
        self.selected_label.pack(anchor="w", padx=5, pady=(0, 10))

        summary_frame = ttk.Labelframe(right_panel, text="Performans Özeti", style="White.TLabelframe")
        summary_frame.pack(fill="x", padx=5, pady=10)

        self.toxic_label = ttk.Label(summary_frame, text="Toksik Cümle Sayısı: 0", style="White.TLabel")
        self.toxic_label.pack(anchor="w", padx=5, pady=2)

        self.non_toxic_label = ttk.Label(summary_frame, text="Non-Toksik Cümle Sayısı: 0", style="White.TLabel")
        self.non_toxic_label.pack(anchor="w", padx=5, pady=2)

        # Frame 2: Threshold setting
       # Başlık label'ını değişkene ata ki sonradan güncelleyebilelim
        self.threshold_label = ttk.Label(
            right_panel,
            text=f"Toksiklik Eşiği (%{self.threshold_var.get():.0f}):",
            font=("Segoe UI", 11, "bold"),
            style="White.TLabel"
        )
        self.threshold_label.pack(anchor="w", padx=5, pady=(0, 5))

        # Slider
        self.threshold_slider = ttk.Scale(
            right_panel,
            from_=0,
            to=100,
            variable=self.threshold_var,
            orient="horizontal",
            length=400
        )
        self.threshold_slider.pack(padx=5, fill="x", expand=True)

        # trace ile hem başlığı hem değeri güncelle
        self.threshold_var.trace("w", lambda *args: self.update_threshold_labels())


        # Clear button
        self.btn_clear_text = ttk.Button(right_panel, text="Temizle")
        self.btn_clear_text.pack(pady=5, padx=5, fill="x")

        # Frame 3: Analysis Results
        frame3 = tk.Frame(self.root, bg="#f9f9f9")
        frame3.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(frame3, text="Analiz Sonuçları:", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 10)) # Increased font size and added pady

        self.result_text = ScrolledText(frame3, wrap="word", height=15, font=("Arial", 10), # Slightly increased font size
                                        padx=30, # Add internal padding to the text widget
                                        relief="flat", borderwidth=2, highlightbackground="#cccccc", highlightcolor="#cccccc") # Modern border
        self.result_text.pack(fill="both", expand=True)


    def set_model_files(self, files):
        if files:
            self.model_combo['values'] = files
            self.model_combo.set(files[0])
            self.model_combo.config(state="readonly")
        else:
            self.model_combo['values'] = ["Model Bulunamadı"]
            self.model_combo.set("Model Bulunamadı")
            self.model_combo.config(state="disabled")
            messagebox.showwarning("Uyarı", "models/DNN Models klasöründe .pt uzantılı model dosyası bulunamadı.") #

    def update_selected_model_label(self, filename, model_type):
        self.selected_label.config(text=f"Seçilen: {filename} ({model_type})")

    def update_summary_labels(self, toxic_count, non_toxic_count):
        self.toxic_label.config(text=f"Toksik Cümle Sayısı: {toxic_count}")
        self.non_toxic_label.config(text=f"Non-Toksik Cümle Sayısı: {non_toxic_count}")

    def clear_all_inputs_and_results(self):
        self.manual_entry.delete(0, tk.END)
        self.youtube_url_entry.delete(0, tk.END)
        self.comment_size_entry.delete(0, tk.END)
        self.comment_size_entry.insert(0, "50")
        self.twitter_url_entry.delete(0, tk.END)
        self.result_text.delete("1.0", tk.END)
        self.threshold_var.set(50)
        self.update_summary_labels(0, 0) # Reset summary counts

    # You'll connect the command functions from the main application logic
    # using methods like self.btn_classify_manual.config(command=some_function)
    # in the main_app.py file.