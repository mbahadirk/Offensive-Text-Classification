import os


def getOptions():
    # models klasörünün yolu
    models_folder = "models"
    options = []
    # models klasöründeki dosyaları listele
    if os.path.exists(models_folder):
        for file_name in os.listdir(models_folder):
            if file_name.endswith("model.pkl"):
                options.append(os.path.splitext(file_name)[0])
    else:
        print(f"{models_folder} klasörü bulunamadı.")
    return options
