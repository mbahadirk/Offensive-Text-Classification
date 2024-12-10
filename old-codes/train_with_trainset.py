import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
keras = tf.keras
layers = keras.layers
Tokenizer = keras.preprocessing.text.Tokenizer


dfTrain = pd.read_csv("../datasets/islenmis_veri10k/train_dataset.csv", header=0)
dfEval = pd.read_csv("../datasets/islenmis_veri10k/validation_dataset.csv", header=0)
dfTest = pd.read_csv("../datasets/islenmis_veri10k/test_dataset.csv", header=0)

print(dfTrain.head())

feature_columns = ['toxic','non-toxic']

dfcTrain = dfTrain.copy()
dfcEval = dfEval.copy()
dfcTest = dfTest.copy()

# 0 means non-toxic and 1 means toxic
df_nonToxic = dfcTrain[dfcTrain['Label'] == 0]
df_toxic = dfcTrain[dfcTrain['Label'] == 1]

print("There are", len(df_toxic), "toxic records.", "and ", len(df_nonToxic), "non toxic records")


print("ratio to toxic: non-toxic records", len(df_toxic) / 27537, ":", len(df_nonToxic) / 27537)

df_nonToxic.head()

# merge dataframes
dfcTrain = pd.concat([df_toxic, df_nonToxic], axis=0)

# shuffle the dataframe
dfcTrain = dfcTrain.sample(frac=1, random_state=42).reset_index(drop=True)
dfcEval = dfcEval.sample(frac=1, random_state=42).reset_index(drop=True)
dfcTest = dfcTest.sample(frac=1, random_state=42).reset_index(drop=True)

dfTrain = dfcTrain.copy()
dfcEval = dfcEval.copy()
dfcTest = dfcTest.copy()

train_features = dfTrain
eval_features = dfcEval
test_features = dfcTest

train_labels = train_features.pop('Label')
eval_labels = eval_features.pop('Label')
test_labels = test_features.pop('Label')
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
tokenizer.fit_on_texts(train_features['Content'])

vocab_size = len(tokenizer.word_index) + 1 # add 1 more, otherwise there will be a out ouf bound index error
vocab_size
# tokenizer.word_index

# convert panda dataframe of labels to numpy array
train_labels = np.array(train_labels)
eval_labels = np.array(eval_labels)
test_labels = np.array(test_labels)

# convert each label to array of categorical values
train_labels = keras.utils.to_categorical(train_labels,2)
eval_labels = keras.utils.to_categorical(eval_labels,2)
test_labels = keras.utils.to_categorical(test_labels,2)

train_features_seq = tokenizer.texts_to_sequences(train_features['Content'])
eval_features_seq = tokenizer.texts_to_sequences(eval_features['Content'])

train_features_seq = keras.utils.pad_sequences(train_features_seq,max_len,padding=padding_type)
eval_features_seq = keras.utils.pad_sequences(eval_features_seq,max_len,padding=padding_type)
len(train_features_seq[0])

train_features_seq = np.array(train_features_seq)
eval_features_seq = np.array(eval_features_seq)
eval_features_seq.shape

num_of_classes = len(feature_columns)

# neural network 01
model1 = keras.Sequential([
    keras.layers.Embedding(vocab_size,32),
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
# epochs = 40
#
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
# history = model1.fit(train_features_seq,train_labels,epochs=epochs,verbose=2,callbacks=[early_stopping])
#
# print(history.history['accuracy'][-1])
#
# plt.plot(history.history['loss'])
# plt.xLabel='epoch'
# plt.yLabel='loss'
# plt.title('model 1 loss against epochs')
# plt.show()
#
# plt.plot(history.history['accuracy'])
# plt.xLabel='epoch'
# plt.yLabel='accuracy'
# plt.title('model 1 accuracy against epochs')
# plt.show()
#
# loss1,acc1 = model1.evaluate(eval_features_seq,eval_labels)
# model1.save('models/2fm-average-pooling-2.h5')


epochs = 40

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

model2.save('models/2fm-LSTM-2.h5')



# Test verisini hazırlama (Test seti için padding işlemi)
test_features_seq = tokenizer.texts_to_sequences(test_features['Content'])
test_features_seq = keras.utils.pad_sequences(test_features_seq, max_len, padding=padding_type)

# Test seti üzerindeki modeli değerlendirme
# Model 1 için test sonuçları
test_loss1, test_acc1 = model1.evaluate(test_features_seq, test_labels, verbose=2)
print(f"Model 1 - Test Loss: {test_loss1}, Test Accuracy: {test_acc1}")

# Model 2 için test sonuçları
test_loss2, test_acc2 = model2.evaluate(test_features_seq, test_labels, verbose=2)
print(f"Model 2 - Test Loss: {test_loss2}, Test Accuracy: {test_acc2}")

# Tahminlerin alınması
# Model 1 tahminleri
predictions1 = model1.predict(test_features_seq)
predicted_classes1 = np.argmax(predictions1, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Model 2 tahminleri
predictions2 = model2.predict(test_features_seq)
predicted_classes2 = np.argmax(predictions2, axis=1)

# Doğruluk oranlarını hesaplama ve görüntüleme
from sklearn.metrics import classification_report, confusion_matrix

# Model 1
print("Model 1 Classification Report:")
print(classification_report(true_classes, predicted_classes1, target_names=['Non-Toxic', 'Toxic']))

print("Model 1 Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes1))

# Model 2
print("Model 2 Classification Report:")
print(classification_report(true_classes, predicted_classes2, target_names=['Non-Toxic', 'Toxic']))

print("Model 2 Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes2))

# Örnek tahminlerin görüntülenmesi
sample_idx = np.random.choice(len(test_features_seq), 5, replace=False)  # Rastgele 5 örnek
for idx in sample_idx:
    print(f"\nTest Sample {idx + 1}: {test_features.iloc[idx]['Content']}")
    print(f"True Label: {'Toxic' if true_classes[idx] == 1 else 'Non-Toxic'}")
    print(f"Model 1 Prediction: {'Toxic' if predicted_classes1[idx] == 1 else 'Non-Toxic'}")
    print(f"Model 2 Prediction: {'Toxic' if predicted_classes2[idx] == 1 else 'Non-Toxic'}")
