import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import seaborn as sns


def load_dataset(path="datasets/turkish_dataset/turkce_cumleler_kokler_corrected_50k.csv", rows=2000):
    df = pd.read_csv(path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'text' in df.columns:
        df = df.drop(columns=['text'])
    df = df.head(rows)
    X = df['roots']
    y = df['label']
    return X, y, df


def classify(X, y, modelType="SVM", vectorizerType="BOW", save=False,test_size =0.3,random_state=42, **params):
    # Eğitim ve test veri setlerini ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state))
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
    elif (vectorizerType == "TFIDF"):
        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
    else:
        print("Yanlis vectorizer adi!")
        return



    # Initialize Model
    classifier = None
    if modelType == "SVM":
        # Create and train the SVM model
        classifier = SVC(**params)
    elif modelType == "LogisticRegression":
        # Logistic Regression modelini oluştur ve eğit
        classifier = LogisticRegression(**params)
    elif modelType == "MultinomialNB":
        # MultinomialNB modeli
        classifier = MultinomialNB(**params)
    elif modelType == "DecisionTreeClassifier":
        # Decision Tree modeli
        classifier = DecisionTreeClassifier(**params)
    elif modelType == "KNeighborsClassifier":
        # K-Neighbors modeli
        classifier = KNeighborsClassifier(**params)
    elif modelType == "RandomForestClassifier":
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
    model_path = f'{save_path}_{train_length}_model.pkl'
    vectorizer_path = f'{save_path}_{train_length}_vectorizer.pkl'
    visualize_classification_report(y_test, y_pred, model_path,model_name=f'{modelType} {vectorizerType}')

    # Modeli kaydetme
    if save:
        with open(model_path, 'wb') as file:
            pickle.dump(classifier, file)

        with open(vectorizer_path, 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

    return y_test, y_pred, classifier


def visualize_classification_report(y_test, y_pred, save_path, model_name):
    # classification_report'un çıktısını al
    report = classification_report(y_test, y_pred, output_dict=True)

    # Precision, Recall ve F1-score metriklerini bir DataFrame'e aktar
    report_df = pd.DataFrame(report).transpose()
    report_df.drop('accuracy', inplace=True)

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(4, 1, height_ratios=[15, 2, 10, 1])  # Isı haritası, bar, confusion matrix için düzen

    # İlk subplot: Isı haritası
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(report_df.iloc[:, :-1], annot=True, fmt=".2f", cmap="YlGn", cbar=True, ax=ax1)
    ax1.set_title(f"Train Report for {model_name}")
    ax1.set_xlabel("Metrics")
    ax1.set_ylabel("Classes")
    ax1.xaxis.set_ticks_position('top')  # X ekseni etiketlerini yukarı taşı
    ax1.xaxis.set_label_position('top')  # X ekseni başlığını yukarı taşı

    # İkinci subplot: Accuracy Progress Bar
    ax2 = fig.add_subplot(gs[1])
    accuracy = accuracy_score(y_test, y_pred)
    ax2.barh(["Accuracy"], [accuracy], color="#19e05b", edgecolor="black", height=0.4)
    ax2.set_xlim(0, 1)  # Progress bar için limitler (0-1 arası)
    ax2.set_xlabel("Accuracy")
    ax2.text(accuracy / 2, 0, f"{accuracy:.2f}", fontsize=12, color="black", weight="bold", ha="center", va="center")
    ax2.set_yticks([])  # Y-ticks kaldırılır (sadece bar görünsün)
    ax2.xaxis.set_label_position('top')  # X ekseni başlığını yukarı taşı

    # Üçüncü subplot: Confusion matrix
    ax3 = fig.add_subplot(gs[2])
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['non_toxic', 'toxic'])
    disp.plot(cmap=plt.cm.Blues, ax=ax3, colorbar=False)  # Colorbar'ı kaldırdık
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("")
    ax3.set_ylabel("")

    # Confusion matrix'in altına toplam veri sayısını ekle
    ax4 = fig.add_subplot(gs[3])
    total_samples = len(y_test)
    ax4.text(0.5, 0.5, f"Total Samples: {total_samples}", fontsize=12, ha="center", va="center")
    ax4.axis("off")  # Bu subplot'un eksenlerini kapatıyoruz

    # Layout ayarı
    plt.tight_layout()
    plt.savefig(f'images/{save_path.strip("models/").strip("pkl")}', bbox_inches='tight')
    plt.show()


def create_matrix(y_test, y_pred, classifier):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
