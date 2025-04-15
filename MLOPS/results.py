import pickle
import yaml
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from get_models import getOptions
from model_hyperparameters import param_types
from train_models import load_dataset, visualize_classification_report


def show_hyperparameters():
    models = getOptions()

    for model in models:
        with open(f'models/{model}.pkl', "rb") as file:
            params = pickle.load(file)

        keys = list(params.get_params().keys())
        p_types = list(param_types.keys())
        print(model)

        for key in keys:
            if '__' in key:
                if key.split("__")[1] in p_types:
                    print('\t',key, params.get_params()[key])


def write_best_parameters():
    models = getOptions()
    with open('../results/best_hyperparameters.yaml', 'w') as file:
        for model in models:
            with open(f'models/{model}.pkl', "rb") as f:
                params = pickle.load(f)

            hyperparams = {key: params.get_params()[key].split("__")[1] for key in params.get_params().keys() if '__' in key and key.split("__")[1] in param_types}
            yaml.dump({model: hyperparams}, file, default_flow_style=False)



def visualize_and_write_test_results(dataset_name, test_size):
    x, y, df = load_dataset(path=f'datasets/turkish_dataset/{dataset_name}.csv', rows=-1)

    # Doğru sıralama ile train-test split işlemi
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    # Handle missing values in X_test before transforming
    X_test = np.where(pd.isna(X_test), '', X_test)

    for model in getOptions():
        with open(f'models/{model}.pkl', "rb") as f:
            params = pickle.load(f)
        with open(f'models/{model.split('_model')[0]}_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        # Vektörleştirme
        X_test_vectorized = vectorizer.transform(X_test)

        # Tahmin yap
        y_pred = params.predict(X_test_vectorized)

        # Accuracy'i yazdır
        print(model, accuracy_score(y_test, y_pred))

        # Görselleştir
        visualize_classification_report(y_test, y_pred, model, model)

        # Sonuçları YAML dosyasına yaz
        results = {model: classification_report(y_test, y_pred, output_dict=True)}

        with open(f'results/{dataset_name}.yaml', 'a+') as file:
            yaml.dump(results, file, default_flow_style=False)



# if __name__ == '__main__':
#     visualize_and_write_test_results('turkce_cumleler_kokler_corrected_50k',0.3)


def plot_model_accuracies(results_file, dataset_name):
    # Load the results from the YAML file
    with open(results_file, 'r') as file:
        results = yaml.load(file, Loader=yaml.FullLoader)

    # Separate models into BOW and TF-IDF
    bow_models = []
    bow_accuracies = []
    tf_models = []
    tf_accuracies = []

    for model, metrics in results.items():
        if '_BOW_' in model:
            bow_models.append(model.split('_model')[0])
            bow_accuracies.append(metrics['accuracy'])
        elif '_TF_' in model:
            tf_models.append(model.split('_model')[0])
            tf_accuracies.append(metrics['accuracy'])

    # Sort models and accuracies in ascending order
    bow_models, bow_accuracies = zip(*sorted(zip(bow_models, bow_accuracies), key=lambda x: x[1]))
    tf_models, tf_accuracies = zip(*sorted(zip(tf_models, tf_accuracies), key=lambda x: x[1]))

    # Plot the accuracies with a color palette
    palette = plt.get_cmap('tab10')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(bow_models, bow_accuracies, color=palette(2), width=0.5)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('BOW Models')
    ax1.set_ylim(0, 1)  # Accuracy ranges from 0 to 1
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.bar(tf_models, tf_accuracies, color=palette(3), width=0.5)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('TF-IDF Models')
    ax2.set_ylim(0, 1)  # Accuracy ranges from 0 to 1
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Add accuracy values on bars
    for ax in [ax1, ax2]:
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{bar.get_height():.2f}', ha='center', fontsize=8)

    plt.suptitle(f'Test Results on {dataset_name} Set')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_model_accuracies('../results/test_results.yaml', 'Test')