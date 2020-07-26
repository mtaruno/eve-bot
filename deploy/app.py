import streamlit as st
import pandas as pd
import numpy as np
import generate_train
import keyword_exploration

# Creating a demo app with multiple pages
def main():
    selected = st.sidebar.radio(
        "Navigate pages", options=["Home", "Generate Train", "Keyword Exploration"]
    )
    print(selected)
    if selected == "Home":
        home()
    elif selected == "Generate Train":
        generate_train.main()
    elif selected == "Keyword Exploration":
        keyword_exploration.main()


def home():
    st.title("Welcome to Enhancing Virtual Engagement with EVE")


if __name__ == "__main__":
    main()
