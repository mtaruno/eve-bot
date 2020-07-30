import random

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)
# Data science
import pandas as pd
import numpy as np
import re
import collections
import yaml

# Loading in objects
train = pd.read_pickle("../objects/train.pkl")

with open(r"../objects/entities.yml") as file:
    entities = yaml.load(file, Loader=yaml.FullLoader)

# with open(r"../objects/sorted_predictions.yml") as file:
#     entities = yaml.load(file, Loader=yaml.FullLoader)

# Making a class to define all the actions to do when you are
class Actions:
    memory = {"hardware": [], "app": []}

    def __init__(self, startup):
        # The initial prompt
        self.startup = startup

    # If greet
    def utter_greet(self):
        # Storing the bank of responses
        return random.choice(
            [
                "Hi! My name is EVE. How may I assist you today?",
                "Hello. How may I be of help?",
            ]
        )

    # If goodbye
    def utter_goodbye(self):
        reaffirm = ["Is there anything else I could help you with?"]
        goodbye = [
            "Thank you for your time. Have a nice day!",
            "Glad I could be of help, have a nice day!",
        ]
        return random.choice(goodbye)

    # Speak to representative
    def link_to_human(self):
        return random.choice(["Alright. Let me direct you to a representative!"])

    def battery(self, entity):
        if entity == "none":
            return random.choice(
                ["What device are you using?", "May I know what device you are using?"]
            )
        else:
            return random.choice(
                [
                    "I'm sorry to hear about there. You can check the battery health in your\
                                  settings. If it is below 75%, please consider getting it replaced at your local apple store"
                ]
            )

    def forgot_pass(self):
        reset_appleid = "https://support.apple.com/en-us/HT201355"
        return f"I'm sorry to hear about that, go to {reset_appleid}"

    def payment(self):
        return random.choice(
            ["Login with your Apple ID and update your payment method"]
        )

    def challenge_robot(self):
        return random.choice(
            [
                "I am EVE, your personal assitant, and I was designed by Matthew to assist you.",
            ]
        )

    def update(self, entity):
        # Affirm hardware
        if entity == "none":
            return random.choice(
                ["What device are you using?", "May I know what device you are using?"]
            )
        elif entity == "macbook pro":
            return random.choice(
                [
                    "Find details on how to update your macbook pro here: https://support.apple.com/en-us/HT201541"
                ]
            )
        else:
            return random.choice(
                [
                    "I'm sorry to hear that the update isn't working for you. Please find more information here: https://support.apple.com/en-us/HT201222"
                ]
            )

    def info(self, entity):
        if entity == "macbook pro":
            return random.choice(
                [
                    "Okay! Right now we have 13 and 16 inch macbook pros. Please find more info here: https://www.apple.com/macbook-pro/"
                ]
            )
        if entity == "ipad":
            return random.choice(["We have a few options for iPads ranging from "])
        if entity == "iphone":
            return random.choice(
                [
                    "Our most latest iPhone model is the iPhone 11. It comes in different model sizes. Please find more info here: https://www.apple.com/iphone/"
                ]
            )
        if entity == "none":
            return random.choice(["What would you like to get info on good sir?"])

    def fallback(self):
        return random.choice(
            [
                "I apologize. I didn't quite understand what you tried to say. Could you rephrase?"
            ]
        )

