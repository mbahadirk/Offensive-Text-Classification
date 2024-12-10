import tkinter as tk
# from polyanna import control_tweet



if __name__ == '__main__':
    # Uygulama penceresini oluştur
    window = tk.Tk()
    window.title("Metin Alma Uygulaması")
    window.geometry("300x200")

    # Global bir değişken tanımla
    saved_text = ""

    # Metni kaydeden işlev
    def save_text():
        global saved_text
        saved_text = entry.get()  # Giriş alanından metni al
        # label_result.config(text=f"tweet :\t{control_tweet(saved_text)[0]}\n confidence :\t{control_tweet(saved_text)[1]}\n{control_tweet(saved_text)[2]}")  # Metni etikete yazdır

    # Giriş alanı
    entry_label = tk.Label(window, text="Metninizi girin:")
    entry_label.pack(pady=5)

    entry = tk.Entry(window, width=30)
    entry.pack(pady=5)

    # Test butonu
    test_button = tk.Button(window, text="Test", command=save_text)
    test_button.pack(pady=10)

    # Sonuç etiketi
    label_result = tk.Label(window, text="")
    label_result.pack(pady=10)

    # Pencereyi çalıştır
    window.mainloop()


