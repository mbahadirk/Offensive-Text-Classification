import pickle
from MLOPS import preprocess
import zeyrek

# Zeyrek analizörü oluştur
analyzer = zeyrek.MorphAnalyzer()

# Kullanıcıdan alınan cümle
user_sentence = input("Bir cümle girin: ")

# Preprocess işlemini gerçekleştirin
processed_sentence = preprocess.full_cleaning_pipeline(user_sentence)

# Kaydedilmiş modeli ve vektörleştiriciyi yükle
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Cümleyi vektörleştirin
vectorized_sentence = vectorizer.transform([processed_sentence])

# Tahmin yap
prediction = svm_model.predict(vectorized_sentence)

# Confidence (olasılık) hesaplama
if hasattr(svm_model, "predict_proba"):
    prediction_prob = svm_model.predict_proba(vectorized_sentence)
    confidence = max(prediction_prob[0]) * 100
    print(f"Confidence: %{confidence:.2f}")

# Sonuçları yazdır
label = "Toxic" if prediction[0] == 1 else "Non-Toxic"
print(f"Tahmin: {label}")
