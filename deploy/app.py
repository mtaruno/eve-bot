import streamlit as st
import pandas as pd
import numpy as np
import keyword_exploration
import generate_train
import develop

# Creating a demo app with multiple pages
def main():
    selected = st.sidebar.radio(
        "Navigate pages", options=["Home", "Generate Train", "Keyword Exploration"]
    )
    print(selected)
    if selected == "Home":
        home()
    if selected == "Generate Train":

        def run_generate_train():
            generate_train.main()

        run_generate_train()
    elif selected == "Keyword Exploration":

        def run_keyword_explore():
            keyword_exploration.main()

        run_keyword_explore()


def home():
    st.title("Welcome to Enhancing Virtual Engagement with EVE")
    develop.main()


if __name__ == "__main__":
    main()
