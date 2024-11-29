import pandas as pd
import numpy as np

# CSV dosyasını yükle
input_csv = "datasets/islenmis_veri_2.csv"  # Dosya adını buraya girin
data = pd.read_csv(input_csv)

# Veriyi karıştır (shuffle)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Veri setinin boyutunu hesapla
total_size = len(data)

# Bölme oranlarına göre indeks hesapla
train_end = int(total_size * 0.8)
val_end = int(total_size * 0.98)

# Veriyi böl
train_data = data.iloc[:train_end]
val_data = data.iloc[train_end:val_end]
test_data = data.iloc[val_end:]

# Bölünmüş verileri CSV dosyalarına kaydet
train_data.to_csv("datasets/islenmis_veri10k/train_dataset.csv", index=False)
val_data.to_csv("datasets/islenmis_veri10k/validation_dataset.csv", index=False)
test_data.to_csv("datasets/islenmis_veri10k/test_dataset.csv", index=False)