import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
keras = tf.keras
layers = keras.layers
Tokenizer = keras.preprocessing.text.Tokenizer


df = pd.read_csv("datasets/labeled_data.csv", header=0)

df = df.drop(df.columns[:5], axis=1)

print(df.head())

feature_columns = ['hate_speech','offensive_language','ok']

dfc = df.copy()

df_hate_speech = dfc[dfc['class']==0]
df_offensive = dfc[dfc['class']==1]
df_ok = dfc[dfc['class']==2]

print("There are",len(df_hate_speech), "hate speech records.","and ",len(df_offensive),"offensive records","and",len(df_ok),"are ok")

df_ok.head(20)

print("ratio to hate speech:offensive records:ok are ",len(df_hate_speech)/1430,":",len(df_offensive)/1430,":",len(df_ok)/1430)


df_offensive = df_offensive[5000:14595]
len(df_offensive)

df_offensive.head()

# merge dataframes
dfc = pd.concat([df_hate_speech,df_offensive,df_ok], axis=0)

# shuffle the dataframe
dfc = dfc.sample(frac=1, random_state=42).reset_index(drop=True)
dfc.head()

dfc.shape

df = dfc.copy()
num_of_rows = len(df.index)
p80 = round(num_of_rows * 0.8)
p99 = round(num_of_rows * 0.99)

train_features = df[0:p80]
eval_features = df[p80:p99]
test_features = df[p99:]

train_labels = train_features.pop('class')
eval_labels = eval_features.pop('class')
test_labels = test_features.pop('class')
print(train_features.shape)
print(eval_features.shape)
print(test_features.shape)

train_labels.head()

# encoding words
padding_type = 'pre'
max_len = 60
trun_type='post'
oov_token = '<oov>'

tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(train_features['tweet'])

vocab_size = len(tokenizer.word_index) + 1 # add 1 more, otherwise there will be a out ouf bound index error
vocab_size
# tokenizer.word_index

# convert panda dataframe of labels to numpy array
train_labels = np.array(train_labels)
eval_labels = np.array(eval_labels)
test_labels = np.array(test_labels)

# convert each label to array of categorical values
train_labels = keras.utils.to_categorical(train_labels,3)
eval_labels = keras.utils.to_categorical(eval_labels,3)
test_labels = keras.utils.to_categorical(test_labels,3)

train_features_seq = tokenizer.texts_to_sequences(train_features['tweet'])
eval_features_seq = tokenizer.texts_to_sequences(eval_features['tweet'])

train_features_seq = keras.utils.pad_sequences(train_features_seq,max_len,padding=padding_type)
eval_features_seq = keras.utils.pad_sequences(eval_features_seq,max_len,padding=padding_type)
len(train_features_seq[0])

train_features_seq = np.array(train_features_seq)
eval_features_seq = np.array(eval_features_seq)
eval_features_seq.shape

num_of_classes = len(feature_columns)

# neural network 01
model1 = keras.Sequential([
    keras.layers.Embedding(vocab_size,32,input_length=max_len),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(num_of_classes, activation='softmax')
])

# neural network 02
model2 = keras.Sequential([
    keras.layers.Embedding(vocab_size,32,input_length=max_len),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(num_of_classes, activation='softmax')
])

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model1.summary())
epochs = 200

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
history = model1.fit(train_features_seq,train_labels,epochs=epochs,verbose=2,callbacks=[early_stopping])

print(history.history['accuracy'][-1])

plt.plot(history.history['loss'])
plt.xLabel='epoch'
plt.yLabel='loss'
plt.title('model 1 loss against epochs')
plt.show()
plt.savefig('model1_loss.png')

plt.plot(history.history['accuracy'])
plt.xLabel='epoch'
plt.yLabel='accuracy'
plt.title('model 1 accuracy against epochs')
plt.show()
plt.savefig('images/model1_accuracy.png')

loss1,acc1 = model1.evaluate(eval_features_seq,eval_labels)
model1.save('images/2fm-average-pooling.h5')


epochs = 200

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model2_history = model2.fit(train_features_seq,train_labels,epochs=epochs,verbose=2,callbacks=[early_stopping])

plt.plot(model2_history.history['loss'])
plt.xLabel='epoch'
plt.yLabel='loss'
plt.title('model 2 loss against epochs')
plt.show()

plt.plot(model2_history.history['accuracy'])
plt.xLabel='epoch'
plt.yLabel='accuracy'
plt.title('model 2 accuracy against epochs')
plt.show()

loss2,acc2 = model2.evaluate(eval_features_seq,eval_labels)



# reset the index of test_features and labels
test_features = test_features.reset_index(drop=True)
test_features.head()

model2.save('2fm-LSTM.h5')

model = keras.models.load_model('2fm-LSTM.h5')

def manual_test(index):
  encoded = tokenizer.texts_to_sequences([test_features.loc[index]['tweet']])
  encoded = keras.utils.pad_sequences(encoded,60,padding='pre',truncating='post')
  preds = model.predict(encoded)
  pred = preds[-1]
  classification = feature_columns[np.argmax(pred)]
  confidence = np.max(pred) *100
  print("tweet is",test_features.loc[index]['tweet'],"\nprediction/classification is",classification,".confidence",confidence,"%")

index = input("pick a index below 152: ")
manual_test(int(index))

import json

with open('toxic_comment_detection_word_index.json', 'w') as file:
    json.dump(tokenizer.word_index, file)

# Open the file containing the word index
with open('toxic_comment_detection_word_index.json', 'r') as file:
    data = json.load(file)

# Convert the JSON data to a dictionary
word_index = dict(data)


feature_columns = ['hate_speech','offensive_language','ok']

my_tweet = "I don't agree with you. I think your ideas dumb"

oov_token = '<oov>'

tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.word_index = word_index
encoded = tokenizer.texts_to_sequences([my_tweet])
encoded = keras.utils.pad_sequences(encoded,60,padding='pre',truncating='post')

pred = model.predict(encoded)

classification = feature_columns[np.argmax(pred[-1])]
confidence = np.max(pred[-1]) *100
# if pred doesn't have at least 90% confidence I will consider as ok.
if confidence < 90:
  classification = 'ok'
print("\nprediction/classification is",classification)