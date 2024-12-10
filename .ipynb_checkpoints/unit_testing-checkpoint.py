from zeyrek import MorphAnalyzer
import pandas as pd
import nltk #bu ve bunun altındaki satırı punkt kütüphaneniz yüklü ise kapatabilirsiniz.

zeyrek = MorphAnalyzer()

dataset = pd.read_csv('datasets/turkish_dataset/just_testing.csv')

text_zeyrek = []
text = dataset['text']

for text in dataset['text']:
    for index in range(len(text)):
        result = zeyrek.lemmatize(text[index])

    for word_result in result:
        text_zeyrek.append(word_result[1][0])

print('islem tamamlandi')

text_zeyrek.to_csv("processed_toxic_turkish_language.csv", index=False)