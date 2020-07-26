import streamlit as st
from actions import Actions
from ner import extract_app, extract_hardware
from initialize_intent_classification import infer_intent
import pandas as pd
import numpy as np
import yaml

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Loading in entities
with open(r"../objects/entities.yml") as file:
    entities = yaml.load(file, Loader=yaml.FullLoader)

sns.set(style="ticks", color_codes=True)
sns.set_style(style="whitegrid")

st.title("Enhancing Virtual Assistance with EVE")


def main():
    a = Actions()
    a.utter_greet()
    input = st.text_input("Tell EVE something!")
    intents, hardware, app = initialize(input)

    # Initializing dialogue history
    columns = entities["hardware"] + entities["apps"]
    history = pd.DataFrame(dict(zip(columns, np.zeros(len(columns)))))
    st.subheader("Showing Dialogue History")
    st.dataframe(history)


def initialize(user_input):
    """ Takes the user input and returns the entity representation and predicted intent"""
    # Intent classification
    intents = infer_intent(user_input)
    # Further unpacking
    user_input, pred = intents
    pred = {k: round(float(v), 3) for k, v in pred.items()}
    st.subheader("Intent Predictions")

    # Visualizing intent classification
    g = sns.barplot(
        list(pred.keys()),
        list(pred.values()),
        palette=sns.cubehelix_palette(8, reverse=True),
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    st.pyplot(bbox_inches="tight")

    st.subheader("Hardware Identified")
    hardware = extract_hardware(user_input, visualize=True)
    st.text(hardware)

    st.subheader("Applications Identified")
    app = extract_app(user_input, visualize=True)
    st.text(app)

    return (intents, hardware, app)


def reply(pred, hardware, app):
    pass


if __name__ == "__main__":
    main()
