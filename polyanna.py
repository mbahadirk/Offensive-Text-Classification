import tensorflow as tf
import numpy as np
import pandas as pd
import json
keras = tf.keras
layers = keras.layers
Tokenizer = keras.preprocessing.text.Tokenizer
oov_token = '<oov>'
df = pd.read_csv("datasets/islenmis_veri10k/islenmis_veri_2.csv", header=0)



feature_columns = ['toxic','non-toxic']

dfc = df.copy()

# 0 means non-toxic and 1 means toxic
df_nonToxic = dfc[dfc['Label'] == 0]
df_toxic = dfc[dfc['Label'] == 1]

df_nonToxic.head()

# merge dataframes
dfc = pd.concat([df_toxic, df_nonToxic], axis=0)

# shuffle the dataframe
dfc = dfc.sample(frac=1, random_state=42).reset_index(drop=True)

df = dfc.copy()

num_of_rows = len(df.index)
p80 = round(num_of_rows * 0.8)
p99 = round(num_of_rows * 0.99)
tokenizer = Tokenizer(oov_token=oov_token)
train_features = df[0:p80]

tokenizer.fit_on_texts(train_features['Content'])

model = keras.models.load_model('models/2fm-LSTM-2.h5')


with open('toxic_comment_detection_word_index.json', 'w') as file:
    json.dump(tokenizer.word_index, file)

# Open the file containing the word index
with open('toxic_comment_detection_word_index.json', 'r') as file:
    data = json.load(file)

# Convert the JSON data to a dictionary
word_index = dict(data)


feature_columns = ['non-toxic','toxic']

def control_tweet(tweet):
    tokenizer.word_index = word_index
    encoded = tokenizer.texts_to_sequences([tweet])
    encoded = keras.utils.pad_sequences(encoded,60,padding='pre',truncating='post')

    pred = model.predict(encoded)

    classification = feature_columns[np.argmax(pred[-1])]
    confidence = np.max(pred[-1]) *100
    # if pred doesn't have at least 90% confidence I will consider as ok.
    if confidence < 60:
        print(confidence)
        classification = 'toxic'
    else:
        classification = 'non-toxic'
    print(f"{tweet}\nprediction/classification is {confidence}", classification)
    return tweet,confidence,classification
