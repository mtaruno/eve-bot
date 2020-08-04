import streamlit as st
from actions import Actions
from ner import extract_app, extract_hardware
from initialize_intent_classifier import infer_intent
import pandas as pd
import numpy as np
import yaml

from streamlit import caching

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Loading in entities
with open(r"../objects/entities.yml") as file:
    entities = yaml.load(file, Loader=yaml.FullLoader)

# Loading in predictions
# with open(r"../objects/sorted_predictions.yml") as file:
#     sorted_predictions = yaml.load(file, Loader=yaml.FullLoader)

# Loading in train data
train = pd.read_pickle("../objects/train.pkl")

sns.set(style="ticks", color_codes=True)
sns.set_style(style="whitegrid")

# Response template
respond = lambda response: f"EVE: {response}"


def main(phrase="Tell EVE something!"):

    # Instantiating class object for this conversation
    a = Actions(phrase)

    # st.text(respond(a.utter_greet()))

    intents, user_input, history_df, end = conversation(a)
    print(end)

    if st.sidebar.button("Show backend"):
        backend_dash(intents, user_input, history_df)

    if end == False:
        caching.clear_cache()
        conversation(Actions("Could you please rephrase?"))


def conversation(starter):
    """ Represents one entire flow of a conversation that takes in the Actions 
    object to know what prompt to start with """

    a = starter

    user_input, hardware, app, intents, history_df = talk(prompt=a.startup)

    # Storing current state
    max_intent, extracted_entities = action_mapper(history_df)

    if extracted_entities != []:
        if len(extracted_entities) == 1:
            entity = extracted_entities[0]
            print(f"Found 1 entity: {entity}")
        elif len(extracted_entities) == 2:
            entity = extracted_entities[:2]
            print(f"Found 2 entities: {entity}")
    else:
        entity = None

    end = listener(max_intent, entity, a)

    return (intents, user_input, history_df, end)


def talk(prompt):
    """ Goes through an initiates a conversation and returns:
    
    User_input: string
    Hardware: List of strings containing entities extracted
    App: List of strings containing entities extracted
    Intents: Tuple that can be unpacked to:
        - User_input
        - Predictions: Dictionary containing intents as keys and prediction probabilities (0-1) as values
    History_df: Dialogue state given the input

     """
    user_input = st.text_input(prompt)

    # Intents
    intents, hardware, app = initialize(user_input)
    user_input, prediction = intents

    # Initializing
    columns = entities["hardware"] + entities["apps"] + list(prediction.keys())
    history_df = pd.DataFrame(dict(zip(columns, np.zeros(len(columns)))), index=[0])

    # Converting to dialogue history entry, then appending it to a dataframe
    history_df = history_df.append(to_row(prediction, hardware, app), ignore_index=True)

    return (user_input, hardware, app, intents, history_df)


def listener(max_intent, entity, actions):
    """ Takes in dialogue state and maps that to a response"""

    # Nested function for following up
    def follow_up(prompt="Could you please rephrase?"):
        """ Business logic for following up """

        # Boolean to know if conversation has ended
        end = None

        st.text("Did that solve your problem?")
        yes = st.button("Yes")
        no = st.button("No")

        if yes:
            st.text(respond("Great! Glad I was able to be of service to you!"))
            end = True

        if no:
            # Continues to the next conversation
            end = False
        return end

    # Initializing actions object
    a = actions

    # Initializing end
    end = None

    if max_intent == "greeting":
        st.text(respond(a.utter_greet()))

    elif max_intent == "info":
        st.text(respond(a.info(entity)))
        end = follow_up()
    elif max_intent == "update":
        st.text(respond(a.update(entity)))
        end = follow_up()
    elif max_intent == "forgot_password":
        st.text(respond(a.forgot_pass()))
        end = follow_up()
    elif max_intent == "challenge_robot":
        st.text(respond(a.challenge_robot()))
    elif max_intent == "goodbye":
        st.text(respond(a.utter_goodbye()))
        st.image("images/eve-bye.jpg", width=400)
        st.text("Eve waves you goodbye!")
    elif max_intent == "payment":
        st.text(respond(a.payment()))
        end = follow_up()
    elif max_intent == "speak_representative":
        st.text(respond(a.link_to_human()))
        st.image("images/representative.png")
    elif max_intent == "battery":
        st.text(respond(a.battery(entity)))
        end = follow_up()
    elif max_intent == "fallback":
        st.text(respond(a.fallback()))

    return end


def backend_dash(intents, user_input, history_df):
    """ Visualizes with a dashboard the entire dialogue state of a conversation given state params """
    # Showing predictions
    st.subheader("EVE's Predictions")
    # Further unpacking
    user_input, pred = intents
    pred = {k: round(float(v), 3) for k, v in pred.items()}

    # Visualizing intent classification
    g = sns.barplot(
        list(pred.keys()),
        list(pred.values()),
        palette=sns.cubehelix_palette(8, reverse=True),
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    st.pyplot(bbox_inches="tight")

    # Entities captured
    st.subheader("Hardware Entities Detected")
    hardware = extract_hardware(user_input, visualize=True)
    st.subheader("App Entities Detected")
    app = extract_app(user_input, visualize=True)

    # Showing history
    st.subheader("Dialogue State History")
    st.dataframe(history_df)


def to_row(prediction, hardware, app):
    row = []

    # Hardware
    if hardware == []:
        for i in range(len(entities["hardware"])):
            row.append(0)
    else:
        for entity in entities["hardware"]:
            if hardware[0][0] == entity:
                row.append(1)
            else:
                row.append(0)

    # App
    if app == []:
        for i in range(len(entities["apps"])):
            row.append(0)
    else:
        for entity in entities["apps"]:
            if app[0][0] == entity:
                row.append(1)
            else:
                row.append(0)

    # Prediction - inserting all the probabilities
    for i in prediction.items():
        row.append(i[1])

    # Converting to dataframe
    columns = entities["hardware"] + entities["apps"] + list(prediction.keys())
    df = pd.DataFrame(dict(zip(columns, row)), index=[0])

    return df


def action_mapper(history_df):
    """ Simply maps an history state to:
    
    A max intent: String
    Entities: List of entities extracted
    
    """
    prediction_probs = history_df.iloc[-1:, -len(set(train["Intent"])) :]
    predictions = [
        *zip(list(prediction_probs.columns), list(prediction_probs.values[0]))
    ]

    # Finding the entities
    entities = history_df.iloc[-1:, : -len(set(train["Intent"]))]
    mask = [True if i == 1.0 else False for i in list(entities.values[0])]
    extracted_entities = [b for a, b in zip(mask, list(entities.columns)) if a]

    # Finding the max intent by sorting
    predictions.sort(key=lambda x: x[1])
    # Taking the max
    max_tuple = predictions[-1:]
    # Max intent
    #     print(max_tuple)
    max_intent = max_tuple[0][0]
    #     print(f'max_intent{max_intent}')

    # Fallback if confidence scores aren't high enough

    return (max_intent, extracted_entities)


def initialize(user_input):
    """ Takes the user input and returns the entity representation and predicted intent"""
    # Intent classification
    intents = infer_intent(user_input)

    # NER
    hardware = extract_hardware(user_input)
    app = extract_app(user_input)

    if hardware == []:
        hardware = "none"

    if app == []:
        app = "none"

    return (intents, hardware, app)


if __name__ == "__main__":
    main()
