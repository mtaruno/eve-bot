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
import random

# Loading in objects
train = pd.read_pickle("../objects/train.pkl")

with open(r"../objects/entities.yml") as file:
    entities = yaml.load(file, Loader=yaml.FullLoader)

# Making a class to define all the actions to do when you are
class Actions:
    memory = {'hardware': [], 'app': []}
    def __init__(self):
        pass

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
        return

    # Speak to representative
    def link_to_human(self):
        return random.choice(["Alright. Let me direct you to a representative!"])

    def battery(self, hardware = False):
        if hardware == False:
            return ['What hardware are you using?']
        else:
            return random.choice([''])

    def forgot_pass(self):
        pass

    def payment(self):
        # What hardware?

        return [""]

    def challenge_robot(self):
        return random.choice(
            [
                "You're funny. Of course I am a robot.",
                "Yes, and I was designed by Matthew to assist you.",
            ]
        )

    def update(self):
        # Affirm hardware
        if hardware == None:
            pass
