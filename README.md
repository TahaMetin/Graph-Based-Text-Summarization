# Graph Based Text Summarization
This project is a Python application that uses the PyQt5, gensim, networkx, nltk, matplotlib, numpy, re, collections, and sklearn libraries to implement a graph-based text summarization algorithm.
## Overview
The application takes a document as input, transforms the sentences into a graph structure, and visualizes this graph. Each sentence in the document represents a node in the graph. Semantic relationships between sentences are established, and sentences are scored based on these relationships. The application then generates a summary of the text based on the sentence scores.
## How It Works
The user uploads a document through the desktop application.
The application tokenizes the document into sentences and represents each sentence as a node in a graph.
Semantic relationships between sentences are established using Word Embedding and BERT techniques.
Each sentence is scored based on several parameters, including the presence of proper names, numeric data, words from the title, and 'theme words' determined by TF-IDF value.
The application generates a summary of the text based on the sentence scores.
The summary is displayed in the interface.
The application calculates the ROUGE score to measure the similarity between the generated summary and the actual summary of the text.
## Dependencies
This project uses several Python libraries. You will need to have the following libraries installed:
- PyQt5
- gensim
- networkx
- nltk
- matplotlib
- numpy
- re
- collections
- sklearn
## Usage
To use the application, simply run the Python script and follow the prompts in the desktop application to upload a document and generate a summary.