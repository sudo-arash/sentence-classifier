import os
import re
import numpy as np
import tensorflow as tf
import h5py

# === Load model definition from MATLAB export ===
import classifysentences.model as mdl

# === Load vocabulary ===
with open("vocab.txt", "r", encoding="utf-8") as f:
    vocab_line = f.read().strip()
vocab = vocab_line.split(",")
vocab_index = {word: i for i, word in enumerate(vocab)}

# === Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()

    stop_words = set([
        'the', 'is', 'and', 'of', 'in', 'to', 'for', 'it', 'that', 'a', 'this', 'i', 'was', 'on', 'with', 'as', 'at',
        'by', 'an', 'be', 'have', 'from', 'or', 'are', 'not', 'but'
    ])
    return [word for word in tokens if word not in stop_words]

def encode_bow(text, vocab_index):
    tokens = preprocess(text)
    vec = np.zeros(len(vocab_index), dtype=np.float32)
    for word in tokens:
        if word in vocab_index:
            vec[vocab_index[word]] += 1
    return vec

# === Load MATLAB weights ===
def loadWeights(model, filename=os.path.join("classifysentences", "weights.h5"), debug=False):
    with h5py.File(filename, 'r') as f:
        for g in f:
            if isinstance(f[g], h5py.Group):
                group = f[g]
                layerName = group.attrs['Name']
                numVars = int(group.attrs['NumVars'])
                layerIdx = layerNum(model, layerName)
                layer = model.layers[layerIdx]
                weightList = [0]*numVars
                for d in group:
                    dataset = group[d]
                    shp     = list(map(int, dataset.attrs['Shape']))
                    weightNum = int(dataset.attrs['WeightNum'])
                    weightList[weightNum] = tf.constant(dataset[()], shape=shp)
                for w in range(numVars):
                    layer.variables[w].assign(weightList[w])
                if hasattr(layer, 'finalize_state'):
                    layer.finalize_state()

def layerNum(model, layerName):
    for i, layer in enumerate(model.layers):
        if layer.name == layerName:
            return i
    raise ValueError(f"Layer {layerName} not found in model.")

# === Main prediction function ===
def predict_sentiment(text):
    model = mdl.create_model()
    loadWeights(model)
    bow = encode_bow(text, vocab_index)
    output = model.predict(np.expand_dims(bow, axis=0), verbose=0)
    label = np.argmax(output)
    labels = ["Negative", "Positive"]
    print(f"Input: {text}")
    print(f"Prediction: {labels[label]} (raw output: {output})")

# === Run it ===
if __name__ == "__main__":
    # You can change the sentence below
    test_sentence = "This is absolutely amazing and works great!"
    predict_sentiment(test_sentence)
