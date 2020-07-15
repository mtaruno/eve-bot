# Eve Bot
Enhancing Virtual Engagement with a Twitter Apple Support bot utilizing transformer models.

## Finished/In-Progress
* ~~Choose the dataset~~
* ~~Preprocess data with Tensorflow~~
  * ~~Tokenization and all preprocessing steps~~
* Embed the data in a meaningful way
  * ~~TFIDF~~
  * ~~Bag of words~~
  * Document embedding exploration
  * BERT
  * Fasttext
* Perform Unsupervised Learning to get intent cluster labels
  * ~~K-Means~~
  * DBSCAN
  * LDA
  * Silouhette Evaluation
* Supervised learning for intent classification on unseen data
  * Using my Twitter labels
  * Using my synthetic data

## Todo:
* Finish retrieval based chatbot baseline model
  * Intents dictionary
* Utilize Google's GPUs to train the transformer model
* Minimum Viable Product Retrieval-Based Chatbot
* Natural language generation: Generative-Based Chatbot
* Deploy my chatbot model with Cortex OR create flask application with thoughtfully designed UX

# Resources
### Example end-to-end projects:
* [A Transformer Chatbot Tutorial with TensorFlow 2.0](https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2)
  * [Google Colaboratory](https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb#scrollTo=dYRx7YzCW4bu)
  * [A Transformer Chatbot Tutorial with TensorFlow 2.0](https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html)
* [Messenger Chatbot from Scratch](https://github.com/daoudclarke/chatbot-from-scratch)
* [Cortical.io - Next generation of AI business applications](https://www.cortical.io)
  * [Their benchmarking](https://www.cortical.io/solutions/message-intelligence/message-intelligence-benchmarking/)
* [Guide to Machine Reading Comprehension in Production with AllenNLP](https://towardsdatascience.com/a-guide-to-machine-reading-comprehension-in-production-with-allennlp-c545867bfeb1)
* [Cool DiabloGPT Text Generation from Microsoft](https://huggingface.co/microsoft/DialoGPT-medium?text=Omg+you+are+the+worst+player)

### Natural Language Understanding and Neural Machine Translation
* [DialoGPT - transformers 3.0.0 documentation](https://huggingface.co/transformers/model_doc/dialogpt.html)
* [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

### Intent Classification
* [BERT Word Embeddings](https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b)
* [How Intent Classification Works in NLU](https://botfront.io/blog/how-intent-classification-works-in-nlu)

### Natural Language Generation - Transformer Models
* [Tensorflow's Transformer Tutorial](https://www.tensorflow.org/tutorials/text/transformer)
* [Self-Attention Mechanism](https://medium.com/@Alibaba_Cloud/self-attention-mechanisms-in-natural-language-processing-9f28315ff905)

### UX Design
* [How to build a kickass UX for your Chatbot?](https://blog.chatteron.io/how-to-build-a-kick-ass-ux-for-your-chat-bot-f01b46c551db#.ooj0vyif5)

And shoutout to Scikit-learn documentation.
