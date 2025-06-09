import torch
import torch.nn as nn
from transformers import AutoModel, BertModel, BertTokenizer
import os
import torch.serialization
from tkinter import messagebox

# --- Model Sınıfları ---
class BertClassifier(torch.nn.Module):
  def __init__(self, dropout=0.5):
    super(BertClassifier, self).__init__()

    self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
    self.dropout = torch.nn.Dropout(dropout)

    # Kullandığımız önceden eğilmiş model "base" sınıfına ait bir BERT modelidir. Yani;
    # 12 layers of Transformer encoder, 12 attention heads, 768 hidden size, 110M parameters.
    # 768, BERT-base modelindeki hidden size'yi, 2 ise veri setimizdek  i toplam kategori sayısını temsil ediyor.
    self.linear = torch.nn.Linear(768, 2)
    self.relu = torch.nn.ReLU()

  def forward(self, input_id, mask):
    # _ değişkeni dizideki tüm belirteçlerin gömme vektörlerini içerir.
    # pooled_output değişkeni [CLS] belirtecinin gömme vektörünü içerir.
    # Metin sınıflandırma için polled_output değişkenini girdi olarak kullanmak yeterlidir.

    # Attention mask, bir belirtecin gercek bir kelimemi yoksa dolgu mu olduğunu tanımlar.
    # Eğer gerçek bir kelime ise attention_mask=1, eğer dolgu ise attention_mask=0 olacaktır.
    # return_dict, değeri "True ise" bir BERT modeli tahmin, eğitim veya değerlendirme sırasında ortaya çıkan
    # loss, logits, hidden_states ve attentions dan oluşan bir tuple oluşturacaktır.
    _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    final_layer = self.relu(linear_output)

    return final_layer

# BertLSTMClassifier sınıfını ExtendedBinarySentimentClassifier'dan önce tanımlayın
class BertLSTMClassifier(torch.nn.Module):
    def __init__(self, dropout=0.7):
        super(BertLSTMClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=256, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(256 * 2, 2)  # 2 çünkü Bidirectional LSTM
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask):
        # last_hidden_state: her token için embedding (batch_size, seq_len, hidden_size)
        last_hidden_state, _ = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)

        # LSTM'e sokuyoruz tüm token vektörlerini
        lstm_output, _ = self.lstm(last_hidden_state)  # (batch_size, seq_len, 2*hidden_size)

        # Sadece ilk token ([CLS])'ın LSTM çıktısını al
        cls_lstm_output = lstm_output[:, 0, :]  # İlk token'ın çıktısı

        dropout_output = self.dropout(cls_lstm_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
    

    
class ExtendedBinarySentimentClassifier(nn.Module):
    def __init__(self, feature_extractor_model: BertLSTMClassifier, freeze_pretrained=False):
        super(ExtendedBinarySentimentClassifier, self).__init__()

        self.feature_extractor = feature_extractor_model

        if freeze_pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask):
        if not self.feature_extractor:
            raise RuntimeError("Feature extractor (BertLSTM) is not loaded.")

        with torch.no_grad() if not any(p.requires_grad for p in self.feature_extractor.parameters()) else torch.enable_grad():
            last_hidden = self.feature_extractor.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            lstm_output, _ = self.feature_extractor.lstm(last_hidden)
            cls_token = lstm_output[:, 0, :]

        return self.classifier(cls_token)

# --- Model Yükleme ve Yönetimi ---
class ModelManager:
    def __init__(self, device="cpu"):
        self.model = None
        self.selected_model_path = None
        self.device = torch.device(device)
        self.tokenizer = self._load_tokenizer()
        self.TOKENIZER_AVAILABLE = (self.tokenizer is not None)

        # PyTorch'un custom sınıfları güvenli şekilde yüklemesi için
        torch.serialization.add_safe_globals([BertClassifier, BertLSTMClassifier, BertModel, ExtendedBinarySentimentClassifier])

    def _load_tokenizer(self):
        try:
            tokenizer = BertTokenizer.from_pretrained("../models/embeddings/bert-turkish-tokenizer")
            print("BERT Tokenizer yüklendi.")
            return tokenizer
        except Exception as e:
            print(f"BERT Tokenizer yüklenirken hata: {e}")
            messagebox.showerror("Hata", f"BERT Tokenizer yüklenemedi: {e}\nMetin sınıflandırma devre dışı.")
            return None

    def load_model(self, model_type, update_ui_callback=None):
        model_mapping = {
            "Bert": "Bert_91AC_FINAL.pt",
            "BertLstm": "BertLSTM_91AC_FINAL.pt",
            "ExtendedBertLstm": "FINAL_SA_NO_FREEZE_5EP_93AC.pt"
        }

        file_name = model_mapping.get(model_type)
        if not file_name:
            messagebox.showerror("Hata", f"Geçersiz model türü: {model_type}")
            return

        file_path = os.path.join("models/DNN Models", file_name)
        if not os.path.exists(file_path):
            self.model = None
            self.selected_model_path = None
            if update_ui_callback:
                update_ui_callback("Yok", model_type)
            messagebox.showerror("Hata", f"Model dosyası bulunamadı: {file_path}")
            return

        try:
            if model_type == "Bert":
                self.model = BertClassifier(dropout=0.5)
            elif model_type == "BertLstm":
                self.model = BertLSTMClassifier(dropout=0.5)
            elif model_type == "ExtendedBertLstm":
                base_bert_lstm_model = BertLSTMClassifier(dropout=0.5)
                self.model = ExtendedBinarySentimentClassifier(feature_extractor_model=base_bert_lstm_model, freeze_pretrained=False)

            self.model = self.model.to(self.device)

            state_dict = torch.load(file_path, map_location=self.device)

            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                self.model.load_state_dict(state_dict['state_dict'])
            else:
                self.model.load_state_dict(state_dict)

            self.model.eval()
            self.selected_model_path = file_path
            if update_ui_callback:
                update_ui_callback(file_name, model_type)
            print(f"Model yüklendi: {file_path} ({model_type})")

        except Exception as e:
            self.model = None
            self.selected_model_path = None
            if update_ui_callback:
                update_ui_callback("Yok", model_type)
            messagebox.showerror("Hata", f"Model yüklenirken hata: {e}\nLütfen model dosyasının seçilen model türüyle uyumlu olduğundan emin olun.")
            print(f"Model yüklenirken hata detayları: {e}")

    def get_current_model(self):
        return self.model, self.selected_model_path

    def get_current_tokenizer(self):
        return self.tokenizer, self.TOKENIZER_AVAILABLE