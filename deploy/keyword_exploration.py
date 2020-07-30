import streamlit as st
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)
# Making my visualizations pretty
sns.set_style("whitegrid")
# Combination exploration
import itertools
import yaml

# Loading back processed data
processed = pd.read_pickle("../objects/processed.pkl")
processed["Real Inbound"] = [[i] for i in processed["Real Inbound"]]
processed["Real Outbound"] = [[i] for i in processed["Real Outbound"]]


def main():

    st.title("Visualizing Distribution of Intents in My Processed Twitter Data")

    # select = st.multiselect("Choose your dataset")

    # Inputting the intents

    """
    Keyword Search
    """

    # Search by keywords (single keyword filter)
    keyword = st.text_input("What keyword would you like to explore today?")

    # Seeing what the processed Tweets look like
    filt = [
        (i, j) for i, j in enumerate(processed["Processed Inbound"]) if keyword in j
    ]
    filtered = processed.iloc[[i[0] for i in filt]]

    # Showing how many tweets contain the keyword
    st.text(f"{len(filtered)} Tweets contain the keyword {keyword}")

    st.subheader(f"Here are Tweets that contain the keyword")
    # Showing the keyword filtered dataframe
    pd.set_option("display.max_columns", None)
    st.dataframe(filtered.iloc[:, 0])
    pd.set_option("display.max_columns", None)
    st.dataframe(filtered.iloc[:, 1])
    pd.set_option("display.max_columns", None)
    st.dataframe(filtered.iloc[:, 2])

    """
    Intent Exploration
    """

    st.subheader("Intent Distribution in the Data")

    intents = {
        "update": ["update"],
        "battery": ["battery", "power"],
        "forgot_password": ["password", "account", "login"],
        "repair": ["repair", "fix", "broken"],
        "payment": ["credit", "card", "payment", "pay"],
    }

    st.write(intents)

    def get_key_tweets(series, keywords):
        """ Takes as input the list of keywords and outputs the Tweets that contains at least
        one of these keywords """
        keyword_tweets = []
        for tweet in series:
            # Want to check if keyword is in tweets
            for keyword in keywords:
                if keyword in tweet:
                    keyword_tweets.append(tweet)
        return keyword_tweets

    def to_set(l):
        """ In order to make the Tweets a set to check for intersections, we need
        to make them immutable by making it a tuple because sets only accept immutable
        elements """
        return set([tuple(row) for row in l])

    # Using the function above to visualize the distribution of intents in my dataset
    intent_lengths = [
        len(get_key_tweets(processed["Processed Inbound"], intents[intent]))
        for intent in intents.keys()
    ]
    keyword = pd.DataFrame(
        {"intents": list(intents.keys()), "intent_lengths": intent_lengths}
    ).sort_values("intent_lengths", ascending=False)

    # Visualization
    plt.figure(figsize=(9, 7))
    plt.bar(keyword["intents"], keyword["intent_lengths"], color="#00acee")
    plt.title("Distribution of Intents Using Keyword Searching")
    plt.xlabel("Intent")
    plt.xticks(rotation=90)
    plt.ylabel("Number of Tweets with the Intent Keywords")
    st.pyplot(bbox_inches="tight")

    # Proportions visualization
    plt.figure(figsize=(9, 7))
    plt.bar(
        keyword["intents"], keyword["intent_lengths"] * 100 / 75879, color="#00acee"
    )
    plt.title("Distribution of Intents Using Keyword Searching")
    plt.xlabel("Intent")
    plt.xticks(rotation=90)
    plt.ylabel("Percentage of Tweets with the Intent Keywords")
    st.pyplot(bbox_inches="tight")

    """
    Looking at combinations of intents
    """

    # Initializing all the thresholds for min amount of combination appearances
    thres = [500, 10, 5, 5]

    # Intent Tweets have all the keys, and as the value contains all the tweets that contain that key, as a set
    intent_tweets = {}
    for key in intents.keys():
        intent_tweets[key] = to_set(
            get_key_tweets(processed["Processed Inbound"], intents[key])
        )

    # Iterating through all pairs, and getting how many Tweets intersect between the pair
    keyword_overlaps = {}

    # COMBINATIONS OF 2

    # Each i returns a tuple containing a pair of length r, which in this case is 2
    for i in list(itertools.combinations(list(intents.keys()), 2)):
        a = to_set(intent_tweets[i[0]])
        b = to_set(intent_tweets[i[1]])
        # Inserting pair to dictionary
        keyword_overlaps[f"{i[0]} + {i[1]}"] = len(a.intersection(b))

    # Filtering to just the significant ones, which I define as greater than 100
    combs = []
    counts = []
    for i in keyword_overlaps.items():
        if i[1] > thres[0]:
            combs.append(i[0])
            counts.append(i[1])

    # Visualizing as well
    v = pd.DataFrame({"Combination": combs, "Counts": counts}).sort_values(
        "Counts", ascending=False
    )
    plt.figure(figsize=(9, 6))
    sns.barplot(x=v["Combination"], y=v["Counts"], palette="magma")
    plt.title(f"Combinations of 2 Keywords (At Least {thres[0]} Occurances)")
    plt.xticks(rotation=90)
    st.pyplot(bbox_inches="tight")

    # COMBINATIONS OF 3
    keyword_overlaps = {}

    try:
        # Each i returns a tuple containing a pair of length r, which in this case is 3
        for i in list(itertools.combinations(list(intents.keys()), 3)):
            a = to_set(intent_tweets[i[0]])
            b = to_set(intent_tweets[i[1]])
            c = to_set(intent_tweets[i[2]])
            # Inserting pair to dictionary
            keyword_overlaps[f"{i[0]} + {i[1]} + {i[2]}"] = len(
                a.intersection(b).intersection(c)
            )

        # Filtering to just the significant ones, which I define as greater than 100
        combs = []
        counts = []
        for i in keyword_overlaps.items():
            if i[1] > thres[1]:
                combs.append(i[0])
                counts.append(i[1])

        # Visualizing as well
        v = pd.DataFrame({"Combination": combs, "Counts": counts}).sort_values(
            "Counts", ascending=False
        )
        plt.figure(figsize=(9, 6))
        sns.barplot(x=v["Combination"], y=v["Counts"], palette="magma")
        plt.title(f"Combinations of 3 Keywords (At Least {thres[1]} Occurances)")
        plt.xticks(rotation=90)
        st.pyplot(bbox_inches="tight")
    except ValueError as e:
        st.text(f"Not enough 3-combinations (Thres = {thres[1]})")

    # COMBINATIONS OF 4
    keyword_overlaps = {}

    try:
        # Each i returns a tuple containing a pair of length r, which in this case is 4
        for i in list(itertools.combinations(list(intents.keys()), 4)):
            a = to_set(intent_tweets[i[0]])
            b = to_set(intent_tweets[i[1]])
            c = to_set(intent_tweets[i[2]])
            d = to_set(intent_tweets[i[3]])
            # Inserting pair to dictionary
            keyword_overlaps[f"{i[0]} + {i[1]} + {i[2]} + {i[3]}"] = len(
                a.intersection(b).intersection(c).intersection(d)
            )

        # Filtering to just the significant ones, which I define as greater than 10
        combs = []
        counts = []
        for i in keyword_overlaps.items():
            if i[1] > thres[2]:
                combs.append(i[0])
                counts.append(i[1])

        # Visualizing as well
        v = pd.DataFrame({"Combination": combs, "Counts": counts}).sort_values(
            "Counts", ascending=False
        )
        plt.figure(figsize=(9, 6))
        sns.barplot(x=v["Combination"], y=v["Counts"], palette="magma")
        plt.title(f"Combinations of 4 Keywords (At Least {thres[2]} Occurances)")
        plt.xticks(rotation=90)
        st.pyplot(bbox_inches="tight")
    except ValueError as e:
        st.text(f"Not enough 4-combinations (Thres = {thres[2]})")

    # GROUPS OF 5
    keyword_overlaps = {}

    try:
        # Each i returns a tuple containing a pair of length r, which in this case is 5
        for i in list(itertools.combinations(list(intents.keys()), 5)):
            a = to_set(intent_tweets[i[0]])
            b = to_set(intent_tweets[i[1]])
            c = to_set(intent_tweets[i[2]])
            d = to_set(intent_tweets[i[3]])
            e = to_set(intent_tweets[i[4]])
            # Inserting pair to dictionary
            keyword_overlaps[f"{i[0]} + {i[1]} + {i[2]} + {i[3]} + {i[4]}"] = len(
                a.intersection(b).intersection(c).intersection(d).intersection(e)
            )

        # Filtering to just the significant ones, which I define as greater than 5
        combs = []
        counts = []
        for i in keyword_overlaps.items():
            if i[1] > thres[3]:
                combs.append(i[0])
                counts.append(i[1])

        # Visualizing as well
        v = pd.DataFrame({"Combination": combs, "Counts": counts}).sort_values(
            "Counts", ascending=False
        )
        plt.figure(figsize=(9, 6))
        sns.barplot(x=v["Combination"], y=v["Counts"], palette="magma")
        plt.title(f"Combinations of 5 Keywords (At Least {thres[3]} Occurances)")
        plt.xticks(rotation=90)
        st.pyplot(bbox_inches="tight")
    except ValueError as e:
        st.text(f"Not enough 5-combinations (Thres = {thres[3]})")


if __name__ == "__main__":
    main()
