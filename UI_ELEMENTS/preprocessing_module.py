import pandas as pd # pandas kullanmıyor gibi görünse de, full_cleaning_pipeline apply ile kullanılıyor
import re
import html # clean_html_tags_and_time içinde kullanılıyor

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
    words = text.split()
    filtered_words = [word for word in words if len(word) > 1]
    return " ".join(filtered_words)

# Türkçe karakterleri İngilizce eşdeğerlerine çevirme fonksiyonu
def convert_turkish_characters(text):
    turkish_to_english = str.maketrans("çğıöşüâÇĞİÖŞÜÂ", "cgiosuaCGIOSUA")
    return text.translate(turkish_to_english)

# Ana temizleme pipeline fonksiyonu
def full_cleaning_pipeline(text):
    text = clean_text(text)
    text = remove_single_characters(text)
    text = convert_turkish_characters(text)
    return text

# DataFrame üzerinde ön işleme yapmak için (uygulamanızda doğrudan kullanılmıyor olabilir ama modülde kalabilir)
def preprocess_text(df, contextName='text'):
    if contextName in df.columns:
        df[contextName] = df[contextName].apply(full_cleaning_pipeline)
    else:
        print(f"Uyarı: '{contextName}' sütunu DataFrame'de bulunamadı.")
    return df

# Eğer bu modül doğrudan çalıştırılırsa test etme
if __name__ == '__main__':
    # Örnek kullanım
    sample_text = "Merhaba dünya! Bu bir test cümlesi. @kullanici #etiket www.orneksite.com 12:34 Çook güzeel bir şeeyyy..."
    cleaned_sample = full_cleaning_pipeline(sample_text)
    print(f"Orijinal: {sample_text}")
    print(f"Temizlenmiş: {cleaned_sample}")

    # pandas DataFrame testi (eğer pandas kuruluysa)
    try:
        import pandas as pd
        data = {'text': ["Bu bir @test #tweeti!", "Çok iyi bir gün! www.site.com", "saat 14:30 ve ÇOK mutlu."],
                'label': [0, 1, 0]}
        df_test = pd.DataFrame(data)
        print("\nOrijinal DataFrame:")
        print(df_test)

        df_preprocessed = preprocess_text(df_test.copy(), 'text')
        print("\nİşlenmiş DataFrame:")
        print(df_preprocessed)
    except ImportError:
        print("\nPandas yüklü değil, DataFrame testi atlandı.")