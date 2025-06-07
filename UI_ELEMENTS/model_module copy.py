import torch
import torch.nn as nn
from transformers import AutoModel, BertModel, BertTokenizer # BertTokenizer buraya da taşındı
import os
import torch.serialization # For safe_globals
from tkinter import messagebox # Hata mesajları için

# --- Model Sınıfları ---
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        try:
            # Not: Bu yolun doğru olduğundan ve model dosyalarının bulunduğundan emin olun.
            self.bert = AutoModel.from_pretrained("models/embeddings/bert-turkish-model")
        except Exception as e:
            print(f"BERT model yüklenirken hata: {e}")
            raise e
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2) # İkili sınıflandırma için çıktı katmanı boyutu 2

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output # Logitleri döndür





    def forward(self, input_ids, attention_mask):
        # Modelde parametreler donmuşsa gradient'e izin verme
        with torch.no_grad() if not any(p.requires_grad for p in self.feature_extractor.parameters()) else torch.enable_grad():
            # Feature extractor modelinin içindeki bert ve lstm katmanlarını kullan
            last_hidden = self.feature_extractor.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            lstm_output, _ = self.feature_extractor.lstm(last_hidden)
            cls_token = lstm_output[:, 0, :]  # (batch, 512)

        return self.classifier(cls_token)

class BertLSTMClassifier(torch.nn.Module):
    def __init__(self, dropout=0.7):
        super(BertLSTMClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=256, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(256 * 2, 2)  # 2 çünkü Bidirectional LSTM
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask):
        last_hidden_state, _ = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        lstm_output, _ = self.lstm(last_hidden_state)
        cls_lstm_output = lstm_output[:, 0, :]
        dropout_output = self.dropout(cls_lstm_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
class ExtendedBinarySentimentClassifier(nn.Module):
    def __init__(self, pretrained_model_path=None, freeze_pretrained=False):
        super(ExtendedBinarySentimentClassifier, self).__init__()

        # BertLSTMClassifier'ı doğrudan burada örneklendirin
        self.feature_extractor = BertLSTMClassifier(dropout=0.5)
        if pretrained_model_path:
            try:
                # Güvenli yükleme için özel model sınıflarını izin listesine ekle
                torch.serialization.add_safe_globals([BertClassifier, BertLSTMClassifier, BertModel])
                self.feature_extractor.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
                print(f"Pretrained BertLSTM model yüklendi: {pretrained_model_path}")
            except Exception as e:
                print(f"Pretrained BertLSTM model yüklenirken hata: {e}")
                messagebox.showerror("Hata", f"Önceden eğitilmiş BertLSTM modeli yüklenemedi: {e}")
                self.feature_extractor = None # Yükleme başarısız olursa feature_extractor'ı None yap

        if freeze_pretrained and self.feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # LSTM sonrası gelen 512-dim veriyi sınıflandırıcıya getir
        # (256 * 2 = 512, bidirectional LSTM'den dolayı)
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
# --- Model Yükleme ve Yönetimi ---
class ModelManager:
    def __init__(self, device="cpu"):
        self.model = None
        self.selected_model_path = None
        self.device = torch.device(device)
        self.tokenizer = self._load_tokenizer()
        self.TOKENIZER_AVAILABLE = (self.tokenizer is not None)

    def _load_tokenizer(self):
        try:
            tokenizer = BertTokenizer.from_pretrained("models/embeddings/bert-turkish-tokenizer")
            print("BERT Tokenizer yüklendi.")
            return tokenizer
        except Exception as e:
            print(f"BERT Tokenizer yüklenirken hata: {e}")
            messagebox.showerror("Hata", f"BERT Tokenizer yüklenemedi: {e}\nMetin sınıflandırma devre dışı.")
            return None

    def get_model_files(self):
        model_dir = "models/DNN Models" # Bu dizinin var olduğundan emin olun
        try:
            files = os.listdir(model_dir)
            return [f for f in files if f.endswith('.pt')]
        except FileNotFoundError:
            print(f"Model klasörü bulunamadı: {model_dir}")
            return []
        except Exception as e:
            print(f"Model dosyaları alınırken hata: {e}")
            return []

    def load_model(self, file_name, model_type, update_ui_callback=None):
        if not file_name or file_name == "Model Bulunamadı":
            self.model = None
            self.selected_model_path = None
            if update_ui_callback:
                update_ui_callback("Yok", model_type)
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
            # Güvenli yükleme için özel model sınıflarını izin listesine ekle
            torch.serialization.add_safe_globals([BertClassifier, BertLSTMClassifier, BertModel,ExtendedBinarySentimentClassifier])

            # Seçilen model türüne göre sınıf örneğini oluştur
            if model_type == "bert":
                self.model = BertClassifier(dropout=0.5)
            elif model_type == "bert_lstm":
                self.model = BertLSTMClassifier(dropout=0.7)
            elif model_type == "extended_bert_lstm":
                self.model = ExtendedBinarySentimentClassifier(pretrained_model_path=None)    
            else:
                raise ValueError(f"Geçersiz model türü: {model_type}")

            # Modeli cihaza taşı
            self.model = self.model.to(self.device)

            # state_dict'i yükle
            state_dict = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

            # Modeli değerlendirme moduna al
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