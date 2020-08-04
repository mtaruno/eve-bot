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

# NER
import spacy

print(f"spaCy: {spacy.__version__}")
from spacy import displacy
import random
from spacy.matcher import PhraseMatcher
import plac
from pathlib import Path

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

# Cool progress bars
from tqdm import tqdm_notebook as tqdm

tqdm().pandas()  # Enable tracking of execution progress

import collections
import yaml
import pickle

# Reading back in intents
with open(r"../objects/intents.yml") as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)

# Reading in representative intents
# with open(r'../objects/intents_repr.yml') as file:
#     intents_repr = yaml.load(file, Loader=yaml.FullLoader)

# Cool progress bars
from tqdm import tqdm_notebook as tqdm


# Load spaCy
nlp = spacy.load("en")

# Reading in training data
train = pd.read_pickle("../objects/train.pkl")

print(train.head())
print(f"\nintents:\n{intents}")

# Reading in processed data
processed = pd.read_pickle("../objects/processed.pkl")

# Looks like I have to make my own training data

entities = {
    "hardware": [
        "macbook pro",
        "iphone",
        "iphones",
        "mac",
        "ipad",
        "watch",
        "TV",
        "airpods",
    ],
    "apps": [
        "app store",
        "garageband",
        "books",
        "calendar",
        "podcasts",
        "notes",
        "icloud",
        "music",
        "messages",
        "facetime",
        "catalina",
        "maverick",
    ],
}

# Storing it to YAML file
with open("../objects/entities.yml", "w") as outfile:
    yaml.dump(entities, outfile, default_flow_style=False)

# Read in the data

hardware_train = pd.read_pickle('../objects/hardware_train.pkl')
app_train = pd.read_pickle('../objects/app_train.pkl')

# Preview
print(hardware_train[:5])
print(app_train[:5])

"""
Training Recognizer with SGD
"""

# Now we train the recognizer.
def train_spacy(train_data, iterations):
    nlp = spacy.blank("en")  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)

    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable all pipes other than 'ner' during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()

        train_loss = []

        # Go through the training data N times
        for itn in range(iterations):
            print("Starting iteration " + str(itn))

            # Shuffle training data
            random.shuffle(train_data)

            # Iteration level metrics
            losses = {}
            misalligned_count = 0

            # Iterating through every Tweet
            for text, annotations in train_data:
                try:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses,
                    )
                except ValueError as e:
                    misalligned_count += 1
                    # If it goes here, that means there's a misaligned entity
                    print(f"Ignoring misaligned entity...\n{(text,annotations)}")
                    pass

            # Enable this is you want to track misalliged counts
            #             print(f'-- misalligned_count (iteration {itn}): {misalligned_count}')
            # Documenting the loss
            train_loss.append(losses.get("ner"))
            print(f"losses (iteration {itn}): {losses}")

        # Visualizing the loss
        plt.figure(figsize=(10, 6))
        plt.plot([*range(len(train_loss))], train_loss, color="magenta")
        plt.title("Loss at every iteration")
        plt.xlabel("Iteration Number")
        plt.ylabel("Loss")
        plt.show()

    return nlp


# Error rate is going up for the minimum for the path we are currently walking in
# We choose 20 for iterations, but there's a point where if you do it too many times it forgets the
# stuff it knows now

# Training 1
hardware_nlp = train_spacy(hardware_train, 20)

# Save our trained model into a new directory
hardware_nlp.to_disk("models/hardware_big_nlp")

# Training 2
app_nlp = train_spacy(app_train, 10)

# Save our trained model into a new directory
app_nlp.to_disk("models/app_big_nlp")

# Serializing
pickle.dump(hardware_nlp, open("models/hardware_big_nlp.pkl", "wb"))
pickle.dump(app_nlp, open("models/app_big_nlp.pkl", "wb"))

