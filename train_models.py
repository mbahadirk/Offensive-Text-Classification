import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # Bag of Words için
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import os


def load_dataset(path = "datasets/turkish_dataset/turkce_cumleler_kokler_corrected_50k.csv", rows=2000):
    df = pd.read_csv(path)
    df = df.drop(columns=['id', 'text'])
    df = df.head(rows)
    X = df['roots']
    y = df['label']
    return X, y, df


def classify(X, y, modelType="SVM", vectorizerType="BOW", save=False, **params):
    # Eğitim ve test veri setlerini ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    train_length = len(X)
    # Eksik verileri doldur
    X_train = np.where(pd.isna(X_train), '', X_train)
    X_test = np.where(pd.isna(X_test), '', X_test)

    if (vectorizerType == "BOW"):
        # Bag of Words vektörleştirme
        vectorizer = CountVectorizer()
        # vectorizer = CountVectorizer(max_features=500)   # Bag of Words için CountVectorizer kullanıyoruz
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
    elif (vectorizerType == "TF"):
        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
    else:
        print("Yanlis vectorizer adi!")
        return



    # Initialize Model
    classifier = None
    if (modelType == "SVM"):
        # Create and train the SVM model
        classifier = SVC(**params)
        for k, v in params.items():
            setattr(classifier, k, v)
    elif (modelType == "LogisticRegression"):
        # Logistic Regression modelini oluştur ve eğit
        classifier = LogisticRegression(**params)
        for k, v in params.items():
            setattr(classifier, k, v)
    elif (modelType == "MultinomialNB"):
        # MultinomialNB modeli
        classifier = MultinomialNB(**params)
        for k, v in params.items():
            setattr(classifier, k, v)
    elif (modelType == "DecisionTreeClassifier"):
        # Decision Tree modeli
        classifier = DecisionTreeClassifier(**params)
        for k, v in params.items():
            setattr(classifier, k, v)
    elif (modelType == "KNeighborsClassifier"):
        # K-Neighbors modeli
        classifier = KNeighborsClassifier(**params)
    elif (modelType == "RandomForestClassifier"):
        # Random Forest modeli
        classifier = RandomForestClassifier(**params)

    else:
        print("Yanlis model adi!")
        return

    classifier.fit(X_train_vectorized, y_train)
    # Make predictions
    y_pred = classifier.predict(X_test_vectorized)

    print("\nModel Performansi:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    save_path = f'models/{modelType}_{vectorizerType}'
    model_path = f'{save_path}_model_{train_length}.pkl'
    vectorizer_path = f'{save_path}_vectorizer_{train_length}.pkl'

    # Modeli kaydetme
    if save:
        with open(model_path, 'wb') as file:
            pickle.dump(classifier, file)

        with open(vectorizer_path, 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

    return y_test, y_pred, classifier


def create_matrix(y_test, y_pred, classifier):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()