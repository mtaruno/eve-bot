Welcome to EVE Bot.

==========================================================================================

Here is a summary of what each file contains:

==========================================================================================

- 1. EDA, Wrangling, and Initial Preprocessing
Contains my data exploration, ideation, and preprocessing pipeline.

- 1.1. Intent Clustering, Document Embeddings, and Unsupervised Learning
In this notebook, I take the preprocessed and tokenized data in the previous notebook and try to assign labels for each Tweet in the dataset by using meaningful document embedding methods (so my models can read the data) and unsupervised learning methods such as K-Means, DBScan, and LDA.

- 2. Heuristic Intent Distribution Exploration
This is a further EDA step I employed to comprehensively know more about what intent labels truly exist in my data. Doing it by keyword might prove to be a good baseline way to do this. I build off this idea, and do a heuristic clustering of my intents by trying to minimize intent intersections. I try to boil down with this method to have the most distinct and _mutually exclusive_ sets of intents so that Eve bot will be able to be trained to distinguish these intents.

- 2.1. Getting my NN Training Data with Doc2Vec
This notebook is where I generated my training data using the Doc2Vec document embedding method and made it in a format that is more readily suitable for modeling.

- 3. Intent Classification with Keras
This notebook is to use the Keras package and use their implementation of a Bidirectional Long Short Term Memory (LSTM) Neural Network to build a model capable of classifying intents given a user input.

- 3.1. Named Entity Recognition
The chatbot should also be able, based on the intent, to label entities that it stores in its dialog management so that its replies are more accurate. This notebook is where for every Tweet, I try to extract the entities. More particularly, from the utterance as input, I want the output to be all the entities in that utterance stored in a dictionary.

==========================================================================================

"deploy" directory (this directory is where I compiled everything and was able to create my app using Python scripts)

==========================================================================================

- actions.py
This class enumerates all the actions that a user can do. Given the intent predicted by my Keras model, and the entity extracted by my spaCy NER, it will map it to an action and return a response back to the user.

- app.py
Compiles all the other scripts together.

- bot.py
Contains the chatbot page (i.e. the bulk of the logic to create Eve bot). The other pages can be accessed in app.py.

- generate_train.py
A Streamlit tool (I found the visualizations of Streamlit to be very helpful to generate my training data with Doc2Vec and output it into a file.

- initialize_intent_classifier.py
This is a shorthand version of the intent classifier that involves just loading the model in so that my application can get the model predictions given a user input. We assume that the model is already saved to a file so that we can just load it in.

- intent_classifier.py
This is the full fledged intent classifier that takes the data and actually runs the model steps to achieve the final model output.

- keyword_exploration.py
Created my Streamlit tool to explore intents in my dataset using keywords - or combinations of them.

- ner.py
Named Entity Recognition tool.

==========================================================================================

Saved Objects

==========================================================================================

- For the Twitter raw data, please find it in this link:
https://www.kaggle.com/thoughtvector/customer-support-on-twitter

- For the gloVe 50D embedding layers for my intent classifier, please find it in this link (make sure to get glove.twitter.27B.zip):
https://nlp.stanford.edu/projects/glove/

- Objects Directory
This directory contains all the objects I need to run my chatbot. Important files:

train.pkl: My training data that I used
entities.yml: Contains a dictionary of the entity keywords I used to train my custom NER
history_df.pkl: Contains the dialogue history at every timestep
processed.pkl: Contains the processed data from my first notebook step
labels_grand.pkl: Contains the labels from the K-Means clustering for my dataset for both bag of words and tfidf
processed.pkl: Contains my inbound, outbound, and processed inbound versions of my data after the first initial preprocessing step shown in my first notebook
silouette_scores_d2v: Contains silhouette scores for doc2vec encoded data as a result of K-Means clustering
hardware_train.pkl: The data for the hardware entity to train NER model on
app_train.pkl: The data for the app entity to train NER model on

- Models Directory
This directory contains all the saved models. Important files:

* d2v.model: Contains the doc2vec model I trained on my Twitter data
* hardware_big_nlp.pkl: Contains my NER model for the hardware entity
* app_big_nlp.pkl: Contains my NER model for the app entity
* intent_classification_b.h5: Contains my latest intent_classification model
* hardware_big_nlp and app_big_nlp = these are my saved model outputs of my entity recognizer. However, it's ~500MB so I did not include it in this submission. Not to worry though, it's reproducible by simply doing (it will take a much longer time than you expect, but don't worry, it will eventually generate the required files):

	python generate_ner_models.py

- deploy/images Directory:
Contains the images I use for Eve bot.

- deploy/plots Directory:
Contains the png image training results of the intent_classifier.py module.


==========================================================================================

Running instructions

==========================================================================================

Simply cd into the deploy directory, then do:

	streamlit run app.py

You can navigate to the respective pages you're interested in there.

Notebooks should be run with the standard jupyter notebook command in command line.
