import numpy as np

def sentence_to_vec(sentence, model, dim=300, max_len=20):
    sentence = str(sentence)
    words = sentence.lower().split()
    vecs = []
    for word in words:
        if word in model:
            vecs.append(model[word])
    # Şimdi padding yapacağız
    if len(vecs) < max_len:
        # Eksik kelime kadar sıfır vektör ekle
        vecs.extend([np.zeros(dim)] * (max_len - len(vecs)))
    else:
        # Fazlaysa kırp
        vecs = vecs[:max_len]
    return np.array(vecs)
