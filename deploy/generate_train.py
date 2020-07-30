import streamlit as st

import pandas as pd
import numpy as np

# Word Embeddings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import builtins

# Text
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

# Storing as objects via serialization
from tempfile import mkdtemp
import pickle
import joblib

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

# Directory
import os
import yaml
import collections
import math

## LOADING OBJECTS
processed_inbound = pd.read_pickle("../objects/processed_inbound_extra.pkl")
processed = pd.read_pickle("../objects/processed.pkl")

# Reading back in intents
with open(r"../objects/intents.yml") as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)


if __name__ == "__main":
    main()


def main():

    st.title("Training Data Generator Tool")

    """Making my idealized dataset - generating N Tweets similar to this artificial Tweet
    This will then be concatenated to current inbound data so it can be included in the doc2vec training
    """

    # Version 2 - I realized that keywords might get the job done, and it's less risky to
    # add more words for the association power because it's doc2vec
    ideal = {
        "battery": "battery power",
        "forgot_password": "password account login",
        "payment": "credit card payment pay",
        "update": "update upgrade",
        "info": "info information",
        # "lost_replace": "replace lost gone missing trade",
        "location": "nearest apple location store",
    }

    def add_extra(current_tokenized_data, extra_tweets):
        """ Adding extra tweets to current tokenized data"""

        # Storing these extra Tweets in a list to concatenate to the inbound data
        extra_tweets = pd.Series(extra_tweets)

        # Making string form
        print("Converting to string...")
        string_processed_data = current_tokenized_data.apply(" ".join)

        # Adding it to the data, updating processed_inbound
        string_processed_data = pd.concat([string_processed_data, extra_tweets], axis=0)

        # We want a tokenized version
        tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        #     print('Tokenizing...')
        #     string_processed_data.apply(tknzr.tokenize)
        return string_processed_data

    # Getting the lengthened data
    processed_inbound_extra = add_extra(
        processed["Processed Inbound"], list(ideal.values())
    )

    # Saving updated processed inbound into a serialized saved file
    processed_inbound_extra.to_pickle("../objects/processed_inbound_extra.pkl")
    st.subheader("Processed Inbound Extra")
    st.dataframe(processed_inbound_extra)
    st.text(
        "As you can see, I appended the documents I wanted to find the similarity of in this dataframe,\
    and this is something you need to do before I doc2vec vectorize my data.\
    This is because doc2vec model similarity function only could find similarity\
    among Tweets that already exist in the vectorized data."
    )

    @st.cache
    def train_doc2vec(string_data, max_epochs, vec_size, alpha):
        # Tagging each of the data with an ID, and I use the most memory efficient one of just using it's ID
        tagged_data = [
            TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
            for i, _d in enumerate(string_data)
        ]

        # Instantiating my model
        model = Doc2Vec(
            size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1
        )

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print("iteration {0}".format(epoch))
            model.train(
                tagged_data, total_examples=model.corpus_count, epochs=model.iter
            )
            # Decrease the learning rate
            model.alpha -= 0.0002
            # Fix the learning rate, no decay
            model.min_alpha = model.alpha

        # Saving model
        model.save("../models/d2v.model")
        print("Model Saved")

    if st.button("Train doc2vec"):
        train_doc2vec(processed_inbound_extra, max_epochs=100, vec_size=20, alpha=0.025)

    # Loading in my model
    model = Doc2Vec.load("../models/d2v.model")

    # Storing my data into a list - this is the data I will cluster
    inbound_d2v = np.array(
        [model.docvecs[i] for i in range(processed_inbound_extra.shape[0])]
    )

    if st.button("Save vectorized doc2vec"):
        # Saving
        path = "../objects/inbound_d2v.pkl"
        with open(path, "wb") as f:
            pickle.dump(inbound_d2v, f)
        st.text(f"Saved to {path}")

    st.subheader("Doc2Vec vectorized data")
    st.dataframe(inbound_d2v)
    st.text(f"Shape: {inbound_d2v.shape}")

    """
    Finding tags of ideal Tweets
    """
    # Version 2
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

    ## Just tokenizing all the values of ideal' values to be able to be fed in to matching function
    # intents_repr = dict(zip(ideal.keys(), [tknzr.tokenize(v) for v in ideal.values()]))
    # Pythonic way
    intents_repr = {k: tknzr.tokenize(v) for k, v in ideal.items()}
    print(intents_repr)

    # Saving intents_repr into YAML
    with open("../objects/intents_repr.yml", "w") as outfile:
        yaml.dump(intents_repr, outfile, default_flow_style=False)

    # Tags for my dictionary
    tags = []

    tokenized_processed_inbound = processed_inbound.apply(tknzr.tokenize)
    # Find the index locations of specific Tweets

    def report_index_loc(tweet, intent_name):
        """ Takes in the Tweet to find the index for and returns a report of that Tweet index along with what the 
        representative Tweet looks like"""
        try:
            tweets = []
            for i, j in enumerate(tokenized_processed_inbound):
                if j == tweet:
                    tweets.append((i, True))
                else:
                    tweets.append((i, False))
            index = []
            for i in tweets:
                if i[1] == True:
                    index.append(i[0])

            preview = processed_inbound.iloc[index]

            # Appending to indexes for dictionary
            tags.append(str(index[0]))
        except IndexError:
            print("Index not in list, move on")
            return

        return intent_name, str(index[0]), preview

    # Reporting and storing indexes with the function
    st.text("TAGGED INDEXES TO LOOK FOR")
    for j, i in intents_repr.items():
        try:
            st.text("\n{} \nIndex: {}\nPreview: {}".format(*report_index_loc(i, j)))
        except Exception as e:
            st.text("Index ended")

    # Pythonic way of making new dictionary from 2 lists
    intents_tags = dict(zip(intents_repr.keys(), tags))

    st.header("Intents Tags Dictionary")
    st.write(intents_tags)

    """
    ACTUALLY GENERATING MY TRAINING DATA
    """

    ## Getting top n tweets similar to the 0th Tweet
    # This will return the a list of tuples (i,j) where i is the index and j is
    # the cosine similarity to the tagged document index

    # Storing all intents in this dataframe
    train = pd.DataFrame()
    # intent_indexes = {}

    # 1. Adding intent content based on similarity
    def generate_intent(target, itag):
        similar_doc = model.docvecs.most_similar(itag, topn=target)
        # Getting just the indexes
        indexes = [int(i[0]) for i in similar_doc]
        #     intent_indexes[intent_name] = indexes
        # Actually seeing the top 1000 Tweets similar to the 0th Tweet which seems to be about updates
        # Adding just the values, not the index
        # Tokenizing the output
        return [
            word_tokenize(tweet)
            for tweet in list(processed_inbound.iloc[indexes].values)
        ]

    # Updating train data
    for intent_name, itag in intents_tags.items():
        train[intent_name] = generate_intent(1000, itag)

    # 2. Manually added intents
    # These are the remainder intents
    manually_added_intents = {
        "speak_representative": [
            ["talk", "human", "please"],
            ["let", "me", "talk", "to", "apple", "support"],
            ["can", "i", "speak", "agent", "person"],
        ],
        "greeting": [
            ["hi"],
            ["hello"],
            ["whats", "up"],
            ["good", "morning"],
            ["good", "evening"],
            ["good", "night"],
        ],
        "goodbye": [["goodbye"], ["bye"], ["thank"], ["thanks"], ["done"]],
        "challenge_robot": [
            ["robot", "human"],
            ["are", "you", "robot"],
            ["who", "are", "you"],
        ],
    }

    # Inserting manually added intents to data
    def insert_manually(target, prototype):
        """ Taking a prototype tokenized document to repeat until
        you get length target"""
        factor = math.ceil(target / len(prototype))
        print(factor)
        content = prototype * factor
        return [content[i] for i in range(target)]

    # Updating training data
    for intent_name in manually_added_intents.keys():
        train[intent_name] = insert_manually(
            1000, [*manually_added_intents[intent_name]]
        )

    # 3. Adding in the hybrid intents

    hybrid_intents = {
        "update": (
            300,
            700,
            [
                ["want", "update"],
                ["update", "not", "working"],
                ["phone", "need", "update"],
            ],
            intents_tags["update"],
        ),
        "info": (
            800,
            200,
            [
                ["need", "information"],
                ["want", "to", "know", "about"],
                ["what", "are", "macbook", "stats"],
                ["any", "info", "next", "release", "?"],
            ],
            intents_tags["info"],
        ),
        "payment": (
            300,
            700,
            [
                ["payment", "not", "through"],
                ["iphone", "apple", "pay", "but", "not", "arrive"],
                ["how", "pay", "for", "this"],
                ["can", "i", "pay", "for", "this", "first"],
            ],
            intents_tags["payment"],
        ),
        "forgot_password": (
            600,
            400,
            [
                ["forgot", "my", "pass"],
                ["forgot", "my", "login", "details"],
                ["cannot", "log", "in", "password"],
                ["lost", "account", "recover", "password"],
            ],
            intents_tags["forgot_password"],
        ),
    }

    def insert_hybrid(manual_target, generated_target, prototype, itag):
        return insert_manually(manual_target, prototype) + list(
            generate_intent(generated_target, itag)
        )

    # Updating training data
    for intent_name, args in hybrid_intents.items():
        train[intent_name] = insert_hybrid(*args)

    # 4. Converting to long dataframe from wide that my NN model can read in for the next notebook - and wrangling
    neat_train = (
        pd.DataFrame(train.T.unstack())
        .reset_index()
        .iloc[:, 1:]
        .rename(columns={"level_1": "Intent", 0: "Utterance"})
    )
    # Reordering
    neat_train = neat_train[["Utterance", "Intent"]]

    # 5. Saving this raw training data into a serialized file
    neat_train.to_pickle("../objects/train.pkl")

    # Styling display
    show = (
        lambda x: x.style.set_properties(
            **{
                "background-color": "black",
                "color": "lawngreen",
                "border-color": "white",
            }
        )
        .applymap(lambda x: f"color: {'lawngreen' if isinstance(x,str) else 'red'}")
        .background_gradient(cmap="Blues")
    )

    st.header("Training data - Comparing different intents view")
    st.dataframe(show(train))

    st.header("Training data in format to feed into models")
    st.dataframe(show(neat_train))

    """
    INTENT EVALUATION
    """

    st.subheader("Looking at top words at each intent")
    # Storing word rank table dataframes in this dict
    wordranks = {}

    # For visualizing top 10
    def top10_bagofwords(data, output_name, title):
        """ Taking as input the data and plots the top 10 words based on counts in this text data"""
        bagofwords = CountVectorizer()
        # Output will be a sparse matrix
        inbound = bagofwords.fit_transform(data)
        # Inspecting of often contractions and colloquial language is used
        word_counts = np.array(np.sum(inbound, axis=0)).reshape((-1,))
        words = np.array(bagofwords.get_feature_names())
        words_df = pd.DataFrame({"word": words, "count": word_counts})
        words_rank = words_df.sort_values(by="count", ascending=False)
        wordranks[output_name] = words_rank
        # words_rank.to_csv('words_rank.csv') # Storing it in a csv so I can inspect and go through it myself
        # Visualizing top 10 words
        plt.figure(figsize=(12, 6))
        sns.barplot(
            words_rank["word"][:10],
            words_rank["count"][:10].astype(str),
            palette="inferno",
        )
        plt.title(title)

        # Saving
        # plt.savefig(f'visualizations/next_ver/{output_name}.png')
        st.pyplot()

    # Doing my bucket evaluations here - seeing what each distinct bucket intent means
    for i in train.columns:
        top10_bagofwords(
            train[i].apply(" ".join), f"bucket_eval/{i}", f"Top 10 Words in {i} Intent",
        )


if __name__ == "__main__":
    main()

