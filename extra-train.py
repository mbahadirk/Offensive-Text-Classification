import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam


keras = tf.keras
layers = keras.layers
Tokenizer = keras.preprocessing.text.Tokenizer

# Mevcut modeli yükle
loaded_model = tf.keras.models.load_model("models/2fm-average-pooling.h5")

splits = {'train': 'data/train-00000-of-00001-b57a122b095e5ed1.parquet', 'validation': 'data/validation-00000-of-00001-9ea89a9fc1c6b387.parquet', 'test': 'data/test-00000-of-00001-10d11e25d2e9ec6e.parquet'}
df = pd.read_parquet("hf://datasets/badmatr11x/hate-offensive-speech/" + splits["train"])


feature_columns = ['hate_speech','offensive_language','ok']

dfc = df.copy()

df_hate_speech = dfc[dfc['label']==0]
df_offensive = dfc[dfc['label']==1]
df_ok = dfc[dfc['label']==2]

print(df.head)

print("There are",len(df_hate_speech), "hate speech records.","and ",len(df_offensive),"offensive records","and",len(df_ok),"are ok")

df_ok.head(20)

print("ratio to hate speech:offensive records:ok are ",len(df_hate_speech)/3294,":",len(df_offensive)/3294,":",len(df_ok)/3294)





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

train_labels = train_features.pop('label')
eval_labels = eval_features.pop('label')
test_labels = test_features.pop('label')
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






epochs = 200
# loaded_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
fine_tuned_model = loaded_model.fit(train_features_seq, train_labels, epochs=epochs, callbacks=[early_stopping])


plt.plot(fine_tuned_model.history['loss'])
plt.xLabel='epoch'
plt.yLabel='loss'
plt.title('fine-tuned loss against epochs')
plt.show()

plt.plot(fine_tuned_model.history['accuracy'])
plt.xLabel='epoch'
plt.yLabel='accuracy'
plt.title('fine-tuned accuracy against epochs')
plt.show()

loss2,acc2 = loaded_model.evaluate(eval_features_seq,eval_labels)



# reset the index of test_features and labels
test_features = test_features.reset_index(drop=True)
test_features.head()

loaded_model.save('models/fine-tuned-V1.h5')