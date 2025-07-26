# Classify Positive or Negative of Sentences

This project classifies sentences as either **positive** or **negative** using a simple neural network.

The model is trained using the [Sentiment Labelled Sentences Dataset](https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set) from Kaggle, which includes thousands of labeled short sentences from sources like Amazon, Yelp, and IMDB.

## Architecture

The neural network is composed of three key layers:

* `featureInputLayer`: Accepts the bag-of-words representation of the sentence.
* `fullyConnectedLayer(10) + reluLayer`: Extracts key patterns in sentence sentiment.
* `fullyConnectedLayer(2) + softmaxLayer`: Outputs a probability distribution over two classes: positive or negative.

## How to Use the Model?

### In MATLAB (R2024b+ recommended)

1. Load the pretrained model using:

   ```matlab
   load model.mat
   ```
2. Use the model with:

   ```matlab
   testData
   ```

You might need to open the testData file and change several things there.

### In Python (with TensorFlow)

1. Ensure the following files are present:

   * `predict.py`
   * `vocab.txt`
   * `classifysentences/` (folder containing `model.py` and `weights.h5`)
2. Run:

   ```bash
   python predict.py
   ```

You can modify `predict.py` to classify your own input sentence or batch of sentences.


## License & Attribution

This project is **open source**. You are free to:

* Modify it.
* Train it with larger or custom datasets.

However, **you are not allowed to sell** this project or any derivative without permission.

**Credit is required** for:

* This repository.
* The original dataset author on [Kaggle](https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set).


Licensed under **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)** in the name of Arash Amini the author of the project.