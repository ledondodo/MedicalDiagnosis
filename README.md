# Advancing Medical Diagnosis: A Comparative Study of Machine Learning Models and Text Embedding Techniques

## *Class Project 2, Machine Learning CS-433, EPFL*

## :dart: Aim
Our work aims to validate the effectiveness of LLMs by comparing them with various traditional machine learning models and text embedding techniques. Our primary focus is on evaluating these models’ abilities to accurately interpret and analyze medical symptoms, particularly in environments with sparse data.

## :bar_chart: Dataset
The dataset provided by the LiGHT lab at EPFL is confidential but we provide an example of it in our report and all the data transformations we did are availbale in csv files in the TransformedData folder.

## :handshake: Contributors
Arthur Chansel, Marianne Scoglio & Gilles de Waha

This is a private school project.

## :microscope: Project description and results
Recently, Large Language Models have gained attention as a potentially superior solution for medical diagnosis using machine learning, especially in environments with limited healthcare resources. However, LLMs are quite heavy compared to other solutions. Our study explored the efficacy of LLMs by comparing them against various traditional machine learning models and text embedding techniques, focusing on their capability to interpret medical symptoms in data-sparse settings. Our results indicated that both common classification algorithms and neural networks tended to overfit and did not achieve satisfactory F1 score results, a challenge exacerbated by the limited size of our dataset. These findings highlight the potential of LLMs like MEDITRON to enhance performance and robustness in medical diagnostics. Additionally, our approach addressed ethical and privacy considerations related to machine learning in healthcare.

The best model we obtained was Linear SVC with BERT embeddings and it reaches a weighted F1 score of 0.444.

## :card_file_box: Files and contents

`data_transformation.ipynb`: notebook for preprocessing the data, obtaining the five possible data transformations (one-hot encoding, word2vec, doc2vec, TF-IDF and BERT) and saving them in csv files in the TransformedData folder.

`models.ipynb`: notebook for testing various traditional ML models (Random Forest, Linear Support Vector Classifier, Gaussian Naive Bayes...) with the one-hot encoded data and selecting the three best models.

`models_x_data.ipynb`: notebook for testing the three best traditional models found previously and two neural networks (ANN and CNN) with the five different data transformations generated previously.

`helpers.py`: all the helper functions. 

## :thinking: How does it work?
To generate the same results we had, simply run the notebook `models_x_data.ipynb`.


## :book: Requirements
Programming language: Python
-  Python >= 3.6
- Libraries and modules: 
  - numpy
  - pandas
  - matplotlib
  - csv
  - os
  - seaborn
  - sklearn
  - json
  - nltk
  - torch
  - transformers
  - gensim
  - tensorflow

