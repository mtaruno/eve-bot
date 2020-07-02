# Transformer Chatbot
Creating a customer service chatbot with transformer models.

## Finished/In-Progress
* ~~Choose the dataset~~
* ~~Preprocess data with Tensorflow~~
  * Tokenization
* Perform Unsupervised Learning to get intent cluster labels
  * K-Means
  * DBSCAN
  * LDA
* Supervised learning for intent classification on unseen data
* Deploy my chatbot model with Cortex

## Todo:
* Embed the data in a meaningful way
  * Document embedding exploration
* Finish retrieval based chatbot baseline model
  * Intents dictionary
* Utilize Google's GPUs to train the transformer model
* Create flask application and thoughtfully design the UX

# Resources
### Example end-to-end projects:
* [A Transformer Chatbot Tutorial with TensorFlow 2.0](https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2)
  * [Google Colaboratory](https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb#scrollTo=dYRx7YzCW4bu)
  * [A Transformer Chatbot Tutorial with TensorFlow 2.0](https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html)
* [Messenger Chatbot from Scratch](https://github.com/daoudclarke/chatbot-from-scratch)
* [Cortical.io - Next generation of AI business applications](https://www.cortical.io)
  * [Their benchmarking](https://www.cortical.io/solutions/message-intelligence/message-intelligence-benchmarking/)

### Natural Language Understanding and Neural Machine Translation
* [DialoGPT - transformers 3.0.0 documentation](https://huggingface.co/transformers/model_doc/dialogpt.html)
* [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

### Intent Classification
* [BERT Word Embeddings](https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b)

### Natural Language Generation
* [Tensorflow's Transformer Tutorial](https://www.tensorflow.org/tutorials/text/transformer)

### UX Design
* [How to build a kickass UX for your Chatbot?](https://blog.chatteron.io/how-to-build-a-kick-ass-ux-for-your-chat-bot-f01b46c551db#.ooj0vyif5)

And shoutout to Scikit-learn documentation.
