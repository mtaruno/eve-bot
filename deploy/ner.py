# Data science
import pandas as pd
import numpy as np
import sklearn

# NER
import spacy
from spacy import displacy
import random
from spacy.matcher import PhraseMatcher
from pathlib import Path

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)
import collections
import yaml
import pickle
import streamlit as st
import imgkit

# Reading back in intents
with open(r"../objects/intents.yml") as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)

# Reading in training data
train = pd.read_pickle("../objects/train.pkl")

# Reading in processed data
processed = pd.read_pickle("../objects/processed.pkl")

# Read our trained models back in
hardware_nlp = pickle.load(open("../models/hardware_big_nlp.pkl", "rb"))
app_nlp = pickle.load(open("../models/app_big_nlp.pkl", "rb"))

# Wrapper to display my displacy visualizations
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

# Testing out the results
test_text_hardware = "My iphone sucks but my macbook pro doesnt. Why couldnt they make\
            my iphone better. At least I could use airpods with it. Mcabook pro is\
            the best! Apple watches too. Maybe if they made the iphone more like the\
            ipad or my TV it would be alright. Mac. Ugh."
test_text_app = "My top favorite apps include the facetime application, the apple books on my iphone, and the podcasts\
        application. Sometimes instead of spotify I would listen to apple music. My macbook is running\
        Catalina btw."


def extract_hardware(user_input, visualize=False):
    """ Takes as input the user input, and outputs all the entities extracted. Also made a toggler for visualizing with displacy."""
    # Loading it in
    hardware_nlp = pickle.load(open("../models/hardware_big_nlp.pkl", "rb"))
    doc = hardware_nlp(user_input)

    extracted_entities = []

    # These are the objects you can take out
    for ent in doc.ents:
        extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))

    # If you want to visualize
    if visualize == True:
        # Visualizing with displaCy how the document had it's entity tagged (runs a server)
        colors = {"HARDWARE": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        options = {"ents": ["HARDWARE"], "colors": colors}
        # Saves to HTML string
        html = displacy.render(doc, style="ent", options=options)
        # with open("displacy/hardware.html", "a") as out:
        #     out.write(html + "\n")
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    return extracted_entities


def extract_app(user_input, visualize=False):
    """ Takes as input the user input, and outputs all the entities extracted. Also made a toggler for visualizing with displacy."""
    # Loading it in
    app_nlp = pickle.load(open("../models/app_big_nlp.pkl", "rb"))
    doc = app_nlp(user_input)

    extracted_entities = []

    # These are the objects you can take out
    for ent in doc.ents:
        extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))

    # If you want to visualize
    if visualize == True:
        # Visualizing with displaCy how the document had it's entity tagged (runs a server)
        colors = {"APP": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        options = {"ents": ["APP"], "colors": colors}
        html = displacy.render(doc, style="ent", options=options)
        # with open("displacy/hardware.html", "a") as out:
        #     out.write(html + "\n")
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    return extracted_entities


def extract_default(user_input):
    pass


# Test functionality
# print(extract_app(test_text_app))
# extract_hardware(test_text_hardware, visualize=True)
