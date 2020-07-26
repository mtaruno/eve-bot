import streamlit as st
import numpy as np
import pandas as pd
import pickle

"""
Making my streamlit chatbot app
"""
st.title("Welcome to Enhancing Virtual Assistance with EVE Bot!")


@st.cache
def load_pickle(path):
    df = pd.read_pickle(path)
    return df


# Creating a data explorer
train = load_pickle("../objects/train.pkl")
processed = load_pickle("../objects/processed.pkl")


st.sidebar.subheader("Data Toggler")

if st.sidebar.checkbox("Show training data"):
    st.dataframe(train)
    st.dataframe(processed)

history = []

user_input = st.text_input("interact here", "")

temp_user_input = user_input

st.text(user_input)

end_conversation = st.button("END CONVERSATION")

