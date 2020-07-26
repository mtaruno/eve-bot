import re
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load intent classification model
from keras.models import load_model

import pandas as pd
import numpy as np

model = load_model("../models/intent_classification.h5")

# Reading in training data
train = pd.read_pickle("../objects/train.pkl")
# Functions required for initialization
get_max_token_length = lambda series: len(max(series, key=len))


def make_tokenizer(docs, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    t = Tokenizer(filters=filters)
    t.fit_on_texts(docs)
    return t


pad_tweets = lambda encoded_doc, max_length: pad_sequences(
    encoded_doc, maxlen=max_length, padding="post"
)

# Initializing required variables

max_token_length = get_max_token_length(train["Utterance"])
token = make_tokenizer(train["Utterance"])
unique_intents = sorted(list(set(train["Intent"])))


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
    intent_predictions = np.array(model.predict(x)[0])

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
print(conf_dict)
