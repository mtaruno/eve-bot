# Streamlit
import streamlit as st

# Data science
import pandas as pd

print(f"Pandas: {pd.__version__}")
import numpy as np

print(f"Numpy: {np.__version__}")

# Deep Learning
import tensorflow as tf

print(f"Tensorflow: {tf.__version__}")
from tensorflow import keras

print(f"Keras: {keras.__version__}")
import sklearn

print(f"Sklearn: {sklearn.__version__}")

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

import collections
import yaml

# Preprocessing and Keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import re
import os
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Reading back in intents
with open(r"../objects/intents.yml") as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)

# Reading in representative intents
with open(r"../objects/intents_repr.yml") as file:
    intents_repr = yaml.load(file, Loader=yaml.FullLoader)

# Reading in training data
train = pd.read_pickle("../objects/train.pkl")

print(train.head())
print(f"\nintents:\n{intents}")
print(f"\nrepresentative intents:\n{intents_repr}")

"""
KERAS PREPROCESSING
"""
# Function definitions
def make_tokenizer(docs, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    t = Tokenizer(filters=filters)
    t.fit_on_texts(docs)
    return t


encode_tweets = lambda token, words: token.texts_to_sequences(words)

pad_tweets = lambda encoded_doc, max_length: pad_sequences(
    encoded_doc, maxlen=max_length, padding="post"
)

one_hot = lambda encode: OneHotEncoder(sparse=False).fit_transform(encode)

get_max_token_length = lambda series: len(max(series, key=len))


def preprocess():
    # 1. Create tokenizer object
    token = make_tokenizer(train["Utterance"])

    # 2. Finding length of vocabulary
    vocab_size = len(token.word_index) + 1

    # 3. Finding maximum length of Tokens

    max_token_length = get_max_token_length(train["Utterance"])

    print(f"Vocab Size: {vocab_size} \nMax Token Length: {max_token_length}")

    # 4. Encode documents - matching with Keras dictionary

    encoded_tweets = encode_tweets(token, train["Utterance"])

    # 5. Padding my documents - filling with tags to normalize the lengths

    padded_tweets = pad_tweets(encoded_tweets, max_token_length)
    print("Shape of padded tweets:", padded_tweets.shape)
    print("\nPreview of encoded and padded tweets:\n", padded_tweets)

    # 6. One hot encode to represent target variable data (intents)
    # Sorting it to make it consistent every time you initialize this variable anywhere
    unique_intents = sorted(list(set(train["Intent"])))

    # Making another tokenizer
    output_tokenizer = make_tokenizer(
        unique_intents, filters='!"#$%&()*+,-/:;<=>?@[\]^`{|}~'
    )
    encoded_intents = encode_tweets(output_tokenizer, train["Intent"])

    # Reshaping encoded Tweets for this one hot function
    encoded_intents = np.array(encoded_intents).reshape(len(encoded_intents), 1)
    one_hot_intents = one_hot(encoded_intents)
    print(f"\nPreview of intent representation:\n{one_hot_intents}")

    return (
        max_token_length,
        padded_tweets,
        one_hot_intents,
        token,
        unique_intents,
        vocab_size,
    )


(
    max_token_length,
    padded_tweets,
    one_hot_intents,
    token,
    unique_intents,
    vocab_size,
) = preprocess()
print(unique_intents)

# 7. Split in to train and test
X_train, X_val, y_train, y_val = train_test_split(
    padded_tweets,
    one_hot_intents,
    test_size=0.3,
    shuffle=True,
    stratify=one_hot_intents,
)
print(
    f"\nShape checks:\nX_train: {X_train.shape} X_val: {X_val.shape}\ny_train: {y_train.shape} y_val: {y_val.shape}"
)

"""
EMBEDDING MATRIX
"""
# Making my own embedding matrix that's in a specific order
d2v_embedding_matrix = pd.read_pickle("../objects/inbound_d2v.pkl")

# Using gloVe word embeddings
embeddings_index = {}
f = open("../models/glove.twitter.27B/glove.twitter.27B.25d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    embeddings_index[word] = coefs
f.close()

print("Found %s word vectors." % len(embeddings_index))

# Initializing required objects
word_index = token.word_index
EMBEDDING_DIM = 25  # Because we are using the 25D gloVe embeddings

# Getting my embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(embedding_matrix)


def make_model(vocab_size, max_token_length):
    """ In this function I define all the layers of my neural network"""
    # Initialize
    model = Sequential()

    # Adding layers - For embedding layer, I made sure to add my embedding matrix into the weights paramater
    model.add(
        Embedding(
            vocab_size,
            embedding_matrix.shape[1],
            input_length=max_token_length,
            trainable=False,
            weights=[embedding_matrix],
        )
    )
    model.add(Bidirectional(LSTM(128)))
    # Another LSTM layer. If things aren't doing well. Beef up the dense layer size.
    #    model.add(LSTM(128))
    # Try 100
    model.add(
        Dense(600, activation="relu", kernel_regularizer="l2")
    )  # Try 50, another dense layer? This takes a little bit of exploration

    # Adding another dense layer to increase model complexity
    model.add(Dense(600, activation="relu", kernel_regularizer="l2"))

    # Only update 50 percent of the nodes - helps with overfitting
    model.add(Dropout(0.5))

    # This last layer should be the size of the number of your intents!
    # Use sigmoid for multilabel classification, otherwise, use softmax!
    model.add(Dense(10, activation="softmax"))

    return model


# Actually creating my model
model = make_model(vocab_size, max_token_length)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

# Initializing checkpoint settings to view progress and save model
filename = "../models/intent_classification.h5"

# Learning rate scheduling
# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


lr_sched_checkpoint = tf.keras.callbacks.LearningRateScheduler(scheduler)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)


# This saves the best model
checkpoint = ModelCheckpoint(
    filename, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)

# The model you get at the end of it is after 100 epochs, but that might not have been
# the weights most associated with validation accuracy

# Only save the weights when you model has the lowest val loss. Early stopping

# Fitting model
hist = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, lr_sched_checkpoint, early_stopping],
)

"""
Visualizing
"""
# Visualizing Training Loss vs Validation Loss (the loss is how wrong your model is)
plt.figure(figsize=(10, 7))
plt.plot(hist.history["val_loss"], label="Validation Loss")
plt.plot(hist.history["loss"], label="Training Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend()
plt.savefig("plots/intentc_trainval_loss.png")

# Visualizing Testing Accuracy vs Validation Accuracy
plt.figure(figsize=(10, 7))
plt.plot(hist.history["accuracy"], label="Training Accuracy")
plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend()
plt.savefig("plots/intentc_trainval_acc.png")

"""
Model step
"""

# I have to redefine and load in the model saved by my model checkpoint
from keras.models import load_model

model = load_model("../models/intent_classification.h5")


def infer_intent(text):
    """ Takes as input an utterance an outputs a dictionary of intent probabilities """
    # Making sure that my text is a string
    string_text = re.sub(r"[^ a-z A-Z 0-9]", " ", text)

    # Converting to Keras form
    keras_text = token.texts_to_sequences(string_text)

    # Check for and remove unknown words - [] indicates that word is unknown
    if [] in keras_text:
        # Filtering out
        keras_text = list(filter(None, keras_text))
    keras_text = np.array(keras_text).reshape(1, len(keras_text))
    x = pad_tweets(keras_text, max_token_length)

    # Generate class probability predictions
    # You're using the overfit model to predict!
    intent_predictions = np.array(model.predict_proba(x)[0])

    # Match probability predictions with intents
    pairs = list(zip(unique_intents, intent_predictions))
    dict_pairs = dict(pairs)

    # Output dictionary
    output = {
        k: v
        for k, v in sorted(dict_pairs.items(), key=lambda item: item[1], reverse=True)
    }

    return string_text, output


string_text, conf_dict = infer_intent("hi")
print(f"You: {string_text}")
print(f"Eve: \nIntents:{conf_dict}")

