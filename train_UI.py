import tkinter as tk
from tkinter import ttk

from model_hyperparameters import model_names, hyperparameters, param_types
from train_models import load_dataset, classify



def on_model_selected(event):
    model_name.set(model_dropdown.get())
    label_model_info.config(text="Model seçildi")
    if vectorizer_name.get():
        create_hyperparameter_dropdowns()


def on_vectorizer_selected(event):
    vectorizer_name.set(vectorizer_dropdown.get())
    label_vectorizer_info.config(text="Vectorizer seçildi")
    if model_name.get():
        create_hyperparameter_dropdowns()


# Dinamik dropdown menüler için bir çerçeve
hyperparam_frame = None


def create_hyperparameter_dropdowns():
    global hyperparam_frame, hyperparam_dropdowns

    # Önce eski çerçeveyi temizle
    if hyperparam_frame is not None:
        for widget in hyperparam_frame.winfo_children():
            widget.destroy()
        hyperparam_frame.destroy()

    # Hyperparameter dropdownları için sözlüğü sıfırla
    hyperparam_dropdowns = {}

    selected_model = model_name.get()
    if selected_model and selected_model in hyperparameters:
        hyperparam_frame = tk.Frame(root)
        hyperparam_frame.pack(pady=10)

        label_hyperparams = tk.Label(hyperparam_frame, text=f"{selected_model} için Hyperparameters:")
        label_hyperparams.pack()

        params = hyperparameters[selected_model]
        for param, values in params.items():
            frame = tk.Frame(hyperparam_frame)
            frame.pack(pady=5, fill="x")

            label = tk.Label(frame, text=param + ":")
            label.pack(side="left", padx=5)

            dropdown = ttk.Combobox(frame, values=values, state="readonly")
            dropdown.pack(side="left", padx=5)
            dropdown.set(values[0])  # Varsayılan olarak ilk değeri seç
            hyperparam_dropdowns[param] = dropdown


def show_selections_and_call_classifier():
    # Dropdown'dan alınan parametre değerlerini türlerine göre dönüştür
    params = {}
    for param, dropdown in hyperparam_dropdowns.items():
        value = dropdown.get()
        if param in param_types:
            try:
                params[param] = param_types[param](value)  # Tür dönüşümü
            except ValueError:
                result_label.config(
                    text=f"Parametre '{param}' için geçersiz değer: {value}", fg="red"
                )
                return
        else:
            params[param] = value

    # Model ve Vectorizer kontrolü
    if not model_name.get() or not vectorizer_name.get():
        result_label.config(text="Lütfen hem model hem de vectorizer seçin.", fg="red")
        return

    try:
        result_label.config(text="Classifier çalıştırılıyor...", fg="green")
        global X, Y, df
        try:
            X, Y, df = load_dataset(rows=int(rows))
        except:
            print("Veri sayısını seçin")
        classify(X, Y, model_name.get(), vectorizer_name.get(), save=True, **params)
        result_label.config(text="Classifier başarıyla çalıştı!", fg="green")
    except Exception as e:
        result_label.config(text=f"Hata: {str(e)}", fg="red")


# Ana pencere
root = tk.Tk()
root.title("UI Oluşturma")
root.geometry("600x800")

# Alınacak veri sayısı
label_data = tk.Label(root, text="Alınacak Veri Sayısı:")
label_data.pack(pady=5)

data_entry = ttk.Entry(root)
data_entry.pack(pady=5)


def validate_input(P):
    if P.isdigit() or P == "":
        return True
    return False


vcmd = (root.register(validate_input), "%P")
data_entry.config(validate="key", validatecommand=vcmd)


def on_load_dataset():
    global X, Y, df, rows
    rows = data_entry.get()
    if rows:
        X, Y, df = load_dataset(rows=int(rows))
        print('dataset yüklendi')
    else:
        print("Lütfen bir sayı girin.")


load_button = ttk.Button(root, text="Veriyi Yükle", command=on_load_dataset)
load_button.pack(pady=10)

# Model seçimi
label_model = tk.Label(root, text="Model Seç:")
label_model.pack(pady=5)

options = model_names

model_name = tk.StringVar(value="")  # Seçilen modelin adı
model_dropdown = ttk.Combobox(root, values=options, state="readonly")
model_dropdown.pack(pady=5)
model_dropdown.bind("<<ComboboxSelected>>", on_model_selected)

# Model seçildi bilgisi
label_model_info = tk.Label(root, text="", fg="green")
label_model_info.pack(pady=10)

# Vectorizer seçimi
label_vectorizer = tk.Label(root, text="Vectorizer Seç:")
label_vectorizer.pack(pady=5)

vectorizer_options = ["BOW", "TF"]
vectorizer_name = tk.StringVar(value="BOW")  # Seçilen vectorizerin adı
vectorizer_dropdown = ttk.Combobox(root, values=vectorizer_options, state="readonly")
vectorizer_dropdown.pack(pady=5)
vectorizer_dropdown.bind("<<ComboboxSelected>>", on_vectorizer_selected)

label_vectorizer_info = tk.Label(root, text="", fg="green")
label_vectorizer_info.pack(pady=10)

# Hyperparameter dropdownları depolamak için sözlük
hyperparam_dropdowns = {}

# Sonuçları gösteren buton ve etiket
result_button = ttk.Button(root, text="Classifier Çalıştır", command=show_selections_and_call_classifier)
result_button.pack(pady=10)

result_label = tk.Label(root, text="", fg="blue", justify="left")
result_label.pack(pady=10)

root.mainloop()