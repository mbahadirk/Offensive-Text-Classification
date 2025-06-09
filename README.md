# Offensive Text Classification

## Go to Sections
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Pretrained Models](#pretrained-models)
- [Developed Models](#developed-models)

## Project Overview
The primary goal of this project is to develop a **user-friendly desktop application** capable of detecting **toxic content** on social media platforms, particularly **Twitter and YouTube**.

In this project, we combined **trained deep learning models** with an **interactive interface**, allowing users to:
- Manually enter sentences for classification.
- Input Twitter or YouTube links to automatically fetch and analyze the content for toxicity.

---

## Project Objectives
With this application, we aim to:
- Automatically filter toxic comments on social platforms.
- Analyze user reactions containing offensive language.
- Perform accurate classifications on Turkish-language datasets using models specifically trained for the local context.

---

## Technologies Used
- Python  
- PyTorch  
- FastText  
- BERT (Turkish language models)  
- Natural Language Processing (NLP)  
- Tkinter (for GUI development)  

---

## Datasets
We used the following publicly available Turkish datasets for training and evaluation:
- [Turkish Offensive Language Dataset](https://www.kaggle.com/datasets/toygarr/turkish-offensive-language-detection?select=valid.csv)<br>
   The file train.csv contains 42,398, test.csv contains 8,851, valid.csv contains 1,756 annotated tweets.
- [Turkish Sentiment Analysis Dataset](https://www.kaggle.com/datasets/winvoker/turkishsentimentanalysisdataset)
<br>   There are 492.782 labeled sentences. %10 of them were used for testing. 
Positive|54% Notr|35% Negative|12%
---

## Pretrained Models
We fine-tuned the following BERT model for Turkish:
- [dbmdz/bert-base-turkish-uncased](https://huggingface.co/dbmdz/bert-base-turkish-uncased)

---

## Developed Models

| Model Name               | Model Link                                                                                       | Features                              |
|--------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------|
| FastText Classifier      | *[Drive](https://drive.google.com/file/d/1nZpoMegV0iQchQpLMK1GDqFZrBh-kGAR/view?usp=sharing)*    | Lightweight, fast predictions         |
| BERT Fine-tuned Model    | *[Drive](https://drive.google.com/file/d/1lJE72RTfZyrpijcGCNTzt0AFjkpuc07U/view?usp=drive_link)* | High accuracy, contextual analysis    |
| BERT LSTM Model          | *[Drive](https://drive.google.com/file/d/1yPUnhSBdGzhZppXCof8D-te9xK-iXui5/view?usp=drive_link)* | High accuracy, contextual analysis    |
| Extended Bert LSTM Model | *[Drive](https://drive.google.com/file/d/1v3HUdnwxJ3BOcNwqj55tODvsh3iw_5yz/view?usp=drive_link)* | Transfer learned from BERT LSTM model |

---

## Application Features
- ✅ Manual text classification
- ✅ Toxicity detection from Twitter or YouTube links
- ✅ Visual feedback on prediction results
- ✅ Turkish language support
- ✅ Easy-to-use desktop interface

---

## Model Performance

### Accuracy Comparison
![Model Comparisons DNN.png](images%2FModel%20Comparisons%20DNN.png)
### Confusion Matrix
![confussion matrix comparison.png](images%2Fconfussion%20matrix%20comparison.png)

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/mbahadirk/Offensive-Text-Classification
   ```

2. Navigate to the project directory:
   ```bash
   cd Offensive-Text-Classification
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Install the model you want to use with UI. *if you didn't install all models it raises alerts but no problem!* [*go to models*](#developed-models) 
5. Install the required tokenizer:
   ```bash
   python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased'); tokenizer.save_pretrained('./models/embeddings/bert-turkish-tokenizer')"

6. Run the application:
   ```bash
   python UI_ELEMENTS/main_app.py
   ```
   