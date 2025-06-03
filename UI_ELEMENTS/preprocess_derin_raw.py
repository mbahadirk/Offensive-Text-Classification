import pandas as pd
import re
    
"""YAPILANLAR
# Text temizleme fonksiyonu
def clean_text(text):
# Tek harfli kelimeleri kaldırma fonksiyonu
def remove_single_characters(text):
# Tek harfli kelimeleri kaldırma fonksiyonu
def remove_single_characters(text):
#Türkçe karakterleri İngilizce eşdeğerlerine çevirme fonksiyonu
def convert_turkish_characters(text):
"""

#stop words olmayacak
#kök alma olmayacak
#KALDIRILANLAR
# Stop-words kaldırma fonksiyonu
# def remove_stop_words(text):
#     """
#     Stop-words listesindeki kelimeleri metinden kaldırır.
#     """
#     words = text.split()  # Metni kelimelere böl
#     filtered_words = [word for word in words if word not in stop_words]  # Stop-words olmayanları seç
#     return " ".join(filtered_words)  # Kelimeleri birleştir

#kelime dönüştürme 
# def replace_words(text):
#     # Doğrudan eşleştirme yapılacak kelimeler
#     corrections = {
#         'arab': r'\barap\b',
#         'gævur': r'\bgavur\b',
#         'sokim': r'\bsokayım\b',
#         'mına': r'\bamına\b',
#         'türki': r'\btürkiye\b',
#         'salağı': r'\bsalak\b',
#         'kadı': r'\bkadın\b',
#         'amk': r'\baq\b',
#     }
# Kelime normalizasyonu (uzatmaları kaldırma) fonksiyonu
# def normalize_repeated_characters(text):
#     """
#     Tekrarlayan harfleri normalleştirir.
#     Örneğin, 'çoook' -> 'çok', 'çççokk' -> 'çok'
#     """
#     # Harf tekrarlarını azalt (örneğin: "çoook" -> "çok")
#     text = re.sub(r'(.)\1{2,}', r'\1', text)  # Aynı harfin 3 veya daha fazla tekrarını 1'e indir
#     # İki kez tekrar eden harfleri tek bir harfe indir
#     text = re.sub(r'(.)\1', r'\1', text)
#     return text
# def extract_roots(text, analyzer):
#     """
#     Zeyrek ile bir cümlenin köklerine ayrılması.
#     Analiz edilemeyen kelimeler olduğu gibi bırakılır.
#     """
#     if pd.isnull(text):
#         return text

#     words = text.split()
#     root_words = []

#     for word in words:
#         # Zeyrek ile analiz yap
#         analyses = analyzer.analyze(word)
#         if analyses:
#             # İlk analiz sonucunun kökünü al
#             root = analyses[0][0].lemma
#             # Eğer root "Unk" ise orijinal kelimeyi kullan
#             if root == "Unk":
#                 root_words.append(word)
#             else:
#                 root_words.append(root)
#         else:
#             # Analiz edilemeyen kelimeyi olduğu gibi bırak
#             root_words.append(word)

#     return " ".join(root_words)


# Zeyrek analizörü oluştur
# analyzer = zeyrek.MorphAnalyzer()

# stop_words_file = "turkce_stop_words.txt"  # Stop-words dosyasının adı


# Stop-words listesini yükle
# with open(stop_words_file, "r", encoding="utf-8") as f:
#     stop_words = set(word.strip() for word in f.readlines())  # Her bir kelimeyi temizleyip sete ekle

# Text temizleme fonksiyonu
def clean_text(text):
    # 1. @ ve # işaretinden sonraki boşluğa kadar olan kısmı kaldır
    text = re.sub(r"[@#]\S+", "", text)
    # 2. Linkleri kaldır (http ve www ile başlayanlar)
    text = re.sub(r"http\S+|www\S+", "", text)
    # 2. Noktalama işaretlerini kaldır
    text = re.sub(r"[^\w\s]", "", text)
    # 3. Tüm harfleri küçük harfe dönüştür
    text = text.lower()
    # 4. Sadece kelimeler arasında boşluk bırak
    text = re.sub(r"\s+", " ", text).strip()
    return text




# Tek harfli kelimeleri kaldırma fonksiyonu
def remove_single_characters(text):
    words = text.split()  # Metni kelimelere böl
    filtered_words = [word for word in words if len(word) > 1]  # Uzunluğu 1'den büyük olanları seç
    return " ".join(filtered_words)  # Kelimeleri birleştir

# Zeyrek ile kelimeleri köklerine ayırma fonksiyonu
# 

    
    # # Regex ile eşleştirme yapılacak yapılar
    # regex_patterns = {
    #     'lan': r'\bul?[a]*n?\b',  # "lan", "laan", "ula", "la", vb.
    #     'amk': r'\bam[aqk]*\b'    # "amk", "amq", "am","mk","mq", vb.
    # }
    
    # # Doğrudan eşleştirme düzeltmeleri
    # for replacement, pattern in corrections.items():
    #     text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # # Regex temelli eşleştirme düzeltmeleri
    # for replacement, pattern in regex_patterns.items():
    #     text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # return text

#Türkçe karakterleri İngilizce eşdeğerlerine çevirme fonksiyonu
def convert_turkish_characters(text):
    turkish_to_english = str.maketrans("çğıöşüâÇĞİÖŞÜÂ", "cgiosuaCGIOSUA")
    return text.translate(turkish_to_english)

def full_cleaning_pipeline(text):
        text = clean_text(text)  # Metni temizleme
        # text = normalize_repeated_characters(text)  # Tekrarlanan karakterleri normalleştirme
        # text = remove_stop_words(text)  # Stop kelimeleri kaldırma
        text = remove_single_characters(text)  # Tek harfli kelimeleri kaldırma
        # text = extract_roots(text, analyzer)  # Kök ayırma işlemi
        # text = replace_words(text)
        text = convert_turkish_characters(text)  # Türkçe karakter dönüşümü
        return text
    
def preprocess_text(df, contextName = 'text'): 
    # 'text' sütununda tüm işlemleri uygulama
    df[contextName] = df[contextName].apply(full_cleaning_pipeline)
    return df



if __name__ == '__main__':
    # Veri setini oku
    input_file = "datasets/turkish_dataset/test.csv"  # Girdi dosyasının adı
    output_file = "datasets/turkish_dataset/preprocessed_test_turkcekarakterli.csv"   # Çıktı dosyasının adı
    # Veri setini yükle
    df = pd.read_csv(input_file)
    df = df.drop("id",axis=1)
    df = preprocess_text(df)

    # İşlenmiş veri setini kaydet
    df.to_csv(output_file, index=False)

    print(f"Veri seti işlenmiş ve '{output_file}' adlı dosyaya kaydedilmiştir.")