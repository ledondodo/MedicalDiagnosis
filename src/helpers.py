'''
 File name: helpers.py
 Author: ML4Health
 Date created: 05/12/2023
 Date last modified: 20/12/2023
 Python Version: 3.11.4
 '''

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from typing import Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, MaxPooling1D, Conv1D, Flatten
from keras.metrics import F1Score
from keras.optimizers import Adamax, AdamW
from keras.optimizers.legacy import Adam, Adagrad
from keras.regularizers import l2



def load_data(path):
    """Load and concatenate patient data from guideline files into a single DataFrame.

    Parameters:
    path (str): Base directory path containing the 'structured_patients' folder.

    Returns:
    pandas.DataFrame: Consolidated DataFrame of patient data from all guideline files.
    """
    # List of guideline files
    guideline_files = [
        'guideline_usp.jsonl',
        'guideline_rch.jsonl',
        'guideline_nice.jsonl',
        'guideline_mayo.jsonl',
        'guideline_idsa.jsonl',
        'guideline_gc.jsonl',
        'guideline_cma.jsonl',
        'guideline_cdc_diseases.jsonl',
        'guideline_aafp.jsonl'
    ]

    # Loop through each guideline file and load into a DataFrame
    dfs = []
    PATIENTS_FOLDER = path+'structured_patients/'
    for file in guideline_files:
        path = os.path.join(PATIENTS_FOLDER, file)
        df_temp = pd.read_json(path, lines=True)
        dfs.append(df_temp)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    return df

def convert_to_dict(x):
    """Convert structured_patient string into a dictionary
    
    Parameters:
    x (str): A JSON string representing the structured patient data.

    Returns:
    dict: The parsed dictionary if successful, or an empty dictionary if parsing fails.
    """
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return {}
    
def handle_type(df):
    """Convert Dataframe to suitable data types

    Parameters:
    df (pandas.DataFrame): DataFrame with a 'structure' column containing JSON strings.

    Returns:
    pandas.DataFrame: The DataFrame with the 'structure' column converted to dictionaries.
    """
    df['structure'] = df['structure'].apply(convert_to_dict)
    return df

def get_symptoms(df):
    """Extract a Dataframe of patients symptoms from the structured data.

    Parameters:
    df (pandas.DataFrame): DataFrame with a 'structure' column containing patient data.

    Returns:
    pandas.DataFrame: A new DataFrame with each symptom in a separate column.
    """
    symptoms = pd.json_normalize(df['structure'].apply(lambda x: x.get('symptoms', [])))
    symptoms = symptoms.applymap(lambda x: x.get('name of the symptom') if x is not None else None)
    return symptoms

def handle_empty(df, symptoms):
    """Remove patients with no symptoms and no condition
    (identified with 'None', 'none', or None)
    (identified with condition_name='')
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing patient data with a 'condition_name' column.
    symptoms (pandas.DataFrame): DataFrame of patients' symptoms.

    Returns:
    tuple: A tuple containing the updated DataFrame and symptoms DataFrame.
    """
    # Format 'None' and 'none' to None
    symptoms = symptoms.apply(lambda row: None if row[0] in ['None', 'none'] else row, axis=1)
    # Drop None values
    no_symptoms = symptoms.index[symptoms[0].isna()].tolist()
    df = df.drop(no_symptoms)
    symptoms = symptoms.drop(no_symptoms)
    # Drop empty conditions_names
    no_condition = df.index[df.condition_name==''].tolist()
    df = df.drop(no_condition)
    symptoms = symptoms.drop(no_condition)
    return df, symptoms

def one_hot_encoding(df, symptoms):
    """Apply one-hot encoding to the symptoms data

    Parameters:
    df (pandas.DataFrame): DataFrame containing patient data with 'condition_name'.
    symptoms (pandas.DataFrame): DataFrame of patients' symptoms.

    Returns:
    pandas.DataFrame: One-hot encoded DataFrame of symptoms with condition names.
    """
    # One-hot encoding for each symptom
    onehot = pd.get_dummies(symptoms.stack().dropna()).groupby(level=0).max()
    onehot = onehot.astype(int)
    # Delete conditions corresponding to outlier patients
    onehot['condition_name'] = df['condition_name']
    return onehot

def ready_data(onehot):
    """Prepare feature and label sets for modeling from one-hot encoded data.

    Parameters:
    onehot (pandas.DataFrame): One-hot encoded DataFrame with the last column as labels.

    Returns:
    tuple: A tuple containing the features (X) and labels (y).
    """
    X = onehot.iloc[:, :-1]  # Features
    y = onehot.iloc[:, -1]   # Labels
    return X, y

def encode_data(df, symptoms):
    """Encode patient symptoms and condition names for model training.

    Parameters:
    df (pandas.DataFrame): DataFrame containing patient data with 'condition_name'.
    symptoms (pandas.DataFrame): DataFrame of patients' symptoms.

    Returns:
    tuple: A tuple containing the encoded features (X) and labels (y).
    """
    # Encode labels
    encoder = LabelEncoder()
    df["condition_name"] = encoder.fit_transform(df["condition_name"])
    # One-hot encoding for each symptom
    onehot = pd.get_dummies(symptoms.stack().dropna()).groupby(level=0).max()
    onehot = onehot.astype(int)
    onehot['condition_name'] = df['condition_name']
    # Encoded data
    X = onehot.iloc[:, :-1]  # Features
    y = onehot.iloc[:, -1]   # Labels
    return X, y

def split_data(X, y):
    """This function stratifies data based on patient conditions, ensuring each condition is
    represented in both training and testing sets. It aims for a 4/5 train (80%) to 1/5 (20%) test split
    overall, with adjustments for small subgroups.

    Parameters:
    X (pandas.DataFrame): The features of patient data.
    y (pandas.Series): The labels (condition names) for the patient data.

    Returns:
    tuple: A tuple containing two DataFrames - the training set and the testing set. 
           Each set includes both features and labels.
    """
    X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

    # Identify unique conditions
    unique_conditions = y.unique()

    # Split each subgroup based on the condition
    for condition in unique_conditions:
        condition_indices = y[y == condition].index
        X_condition = X.loc[condition_indices]
        y_condition = y.loc[condition_indices]

        # Calculate the number of patients per condition for the test set (rounded down, at least 1)
        patients_per_condition_test = int(np.floor(len(y_condition) / 5))

        # Split the subgroup into train and test sets
        if patients_per_condition_test > 0:
            # Split 80%
            X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_condition, y_condition, test_size=patients_per_condition_test, stratify=y_condition, random_state=42)
        else:
            for i in [3,4]:
                if len(y_condition) == i:
                    # Split 33%, 75%
                    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_condition, y_condition, test_size=1, stratify=y_condition, random_state=42)
            if len(y_condition) < 3:
                # Split 0%
                X_train_sub, X_test_sub, y_train_sub, y_test_sub = X_condition, pd.DataFrame(), y_condition, pd.Series()

        # Concatenate the results to the final train and test sets
        X_train = pd.concat([X_train, X_train_sub])
        X_test = pd.concat([X_test, X_test_sub])
        y_train = pd.concat([y_train, y_train_sub])
        y_test = pd.concat([y_test, y_test_sub])

    # Training and testing sets
    train_set = X_train
    train_set['condition_name'] = y_train
    test_set = X_test
    test_set['condition_name'] = y_test

    # Shuffle sets
    train_set = train_set.sample(frac=1,random_state=42)
    test_set = test_set.sample(frac=1,random_state=42)
    return train_set, test_set

def get_Xy(set):
    """Extract features (X) and labels (y) from a dataset.

    Parameters:
    set (pandas.DataFrame): A DataFrame where the last column is the label and the rest are features.

    Returns:
    tuple: A tuple containing the features (X) as a DataFrame and the labels (y) as a Series.
    """
    X = set.iloc[:,:-1]
    y = set.iloc[:,-1]
    return X, y

def compute_scores(y_true, y_pred):
    """
    Compute accuracy, precision, recall, and F1 score for classification predictions.

    Parameters:
    y_true (pandas.Series/array-like): True labels.
    y_pred (pandas.Series/array-like): Predicted labels.

    Returns:
    tuple: A tuple containing accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1

def score_model(lr, train_set, test_set):
    """
    Train a model and compute metrics for it on training and testing sets.

    Parameters:
    lr (estimator): A machine learning model with fit and predict methods.
    train_set (pandas.DataFrame): Training set including features and labels.
    test_set (pandas.DataFrame): Testing set including features and labels.

    Returns:
    pandas.DataFrame: A DataFrame with accuracy, precision, recall, and F1 scores for both training and testing sets.
    """
    # Train model
    X_train, y_train = get_Xy(train_set)
    X_test, y_test = get_Xy(test_set)
    lr.fit(X_train,y_train)
    # Train score
    y_pred = lr.predict(X_train)
    accuracy, precision, recall, f1 = compute_scores(y_train, y_pred)
    scores = pd.DataFrame({'Train': [accuracy, precision, recall, f1]},
                          index=['accuracy', 'precision', 'recall', 'f1'])
    # Test score
    y_pred = lr.predict(X_test)
    accuracy, precision, recall, f1 = compute_scores(y_test, y_pred)
    scores['Test'] = [accuracy, precision, recall, f1]
    return scores

def splitting(X,y):
    """
    Stratified splitting of data into training and testing sets based on unique conditions.

    Parameters:
    X (pandas.DataFrame): The features of patient data.
    y (pandas.Series): The labels (condition names) for the patient data.

    Returns:
    tuple: A tuple containing two DataFrames - the training set and the testing set, each including both features and labels.
    """
    # Identify unique conditions
    unique_conditions = y.unique()

    # Initialize empty DataFrames for train and test sets
    X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

    # Split each subgroup based on the condition
    for condition in unique_conditions:
        condition_indices = y[y == condition].index
        X_condition = X.loc[condition_indices]
        y_condition = y.loc[condition_indices]

        # Calculate the number of patients per condition for the test set (rounded down, at least 1)
        patients_per_condition_test = max(0, int(np.floor(len(y_condition) / 3)))

        # Split the subgroup into train and test sets
        if patients_per_condition_test > 0:
            X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_condition, y_condition, test_size=patients_per_condition_test, stratify=y_condition, random_state=42)
        else:
            X_train_sub, X_test_sub, y_train_sub, y_test_sub = X_condition, pd.DataFrame(), y_condition, pd.Series()

        # Concatenate the results to the final train and test sets
        X_train = pd.concat([X_train, X_train_sub])
        X_test = pd.concat([X_test, X_test_sub])
        y_train = pd.concat([y_train, y_train_sub])
        y_test = pd.concat([y_test, y_test_sub])


    # Training and testing sets
    train_set = X_train
    train_set['condition_name'] = y_train
    test_set = X_test
    test_set['condition_name'] = y_test

    # Shuffle sets
    train_set = train_set.sample(frac=1,random_state=42)
    test_set = test_set.sample(frac=1,random_state=42)
    return train_set, test_set

def fit_models_get_CV_scores(
    models: list,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    scoring: str or dict,
    model_naming: Callable = lambda model: model.__class__.__name__,
) -> pd.DataFrame:
    """
    Fit multiple models and evaluate them using cross-validation.

    Parameters:
    models (list): List of machine learning models to fit.
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Label vector.
    cv (cross-validation strategy): Cross-validation splitting strategy.
    scoring (str or dict): Single string or a dictionary of scorer names and scoring functions.
    model_naming (Callable): Function to generate model names based on their class.

    Returns:
    pd.DataFrame: DataFrame containing cross-validation scores and fit times for each model and fold.
    """
    step = cv.get_n_splits(X)
    if not isinstance(scoring, dict):
        scoring = {scoring: scoring}
    test_scorings_column_names = [f"test_{scoring_name}" for scoring_name in scoring.keys()]
    train_scorings_column_names = [f"train_{scoring_name}" for scoring_name in scoring.keys()]
    models_CV_scores = pd.DataFrame(
        columns=[
            "model",
            "fold",
            "fit_time",
            *test_scorings_column_names,
            *train_scorings_column_names
        ],
        index=range(len(models) * step),
    )
    for i, model in enumerate(models):
        model_name = model_naming(model)
        print(f"Evaluating model: {model_name}")
        pipe = model
        model_CV_scores = cross_validate(
            estimator=pipe,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )
        for test_scoring_name in test_scorings_column_names:
            models_CV_scores[test_scoring_name].iloc[
                i * step : (i + 1) * step
            ] = model_CV_scores[test_scoring_name]
        for train_scoring_name in train_scorings_column_names:
            models_CV_scores[train_scoring_name].iloc[
                i * step : (i + 1) * step
            ] = model_CV_scores[train_scoring_name]
        models_CV_scores["fit_time"].iloc[i * step : (i + 1) * step] = model_CV_scores[
            "fit_time"
        ]

        models_CV_scores["model"].iloc[i * step : (i + 1) * step] = model_name
        models_CV_scores["fold"].iloc[i * step : (i + 1) * step] = np.arange(step)

    return models_CV_scores

def create_ANN(
    X_train_shape1,
    num_classes,
    lr=0.01,
    optimizer='adam',
    loss='categorical_crossentropy',
    hidden_layers_dims = [
        64, 128
    ],
    metrics=[F1Score("weighted")],
    dropout = 0.5,
    reg = 0.0001
):
    """
    Create an Artificial Neural Network (ANN) model with specified parameters.

    Parameters:
    X_train_shape1 (int): Input shape of the training data.
    num_classes (int): Number of output classes.
    lr (float): Learning rate for the optimizer.
    optimizer (str): Name of the optimizer to use.
    loss (str): Loss function.
    hidden_layers_dims (list): List of neuron counts in the hidden layers.
    metrics (list): List of metrics to be evaluated by the model during training and testing.
    dropout (float): Dropout rate for regularization.
    reg (float): L2 regularization factor.

    Returns:
    keras.models.Sequential: Compiled ANN model.
    """
    if optimizer == 'adam':
         optimizer = Adam(learning_rate=lr)
    elif optimizer == 'adagrad':
         optimizer = Adagrad(learning_rate=lr)
    elif optimizer == 'adamw':
         optimizer = AdamW(learning_rate=lr)
    elif optimizer == 'adamax':
         optimizer = Adamax(learning_rate=lr)

    model = Sequential()
    model.add(Dense(hidden_layers_dims[0], activation = "relu", input_dim=X_train_shape1))
    model.add(Dropout(dropout))
    for i in hidden_layers_dims[1:]:
        model.add(Dense(i, activation = "relu",kernel_regularizer=l2(reg)))
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def create_CNN(
    X_train_shape1, 
    num_classes,
    lr=0.001,
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[F1Score("weighted")],
    n_blocks=1,
    dropout = 0.3,
    reg = 0.001
):
    """
    Create a Convolutional Neural Network (CNN) model with specified parameters.

    Parameters:
    X_train_shape1 (int): Input shape of the training data.
    num_classes (int): Number of output classes.
    lr (float): Learning rate for the optimizer.
    optimizer (str): Name of the optimizer to use.
    loss (str): Loss function.
    metrics (list): List of metrics to be evaluated by the model during training and testing.
    n_blocks (int): Number of convolutional blocks.
    dropout (float): Dropout rate for regularization.
    reg (float): L2 regularization factor.

    Returns:
    keras.models.Sequential: Compiled CNN model.
    """
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate=lr)
    elif optimizer == 'adamw':
        optimizer = AdamW(learning_rate=lr)
    elif optimizer == 'adamax':
        optimizer = Adamax(learning_rate=lr)

    model = Sequential()

    for i in range(n_blocks):

        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_shape1,1), kernel_regularizer=l2(reg)))

        model.add(BatchNormalization())
        # Max pooling layer
        model.add(MaxPooling1D(pool_size=2))

        model.add(Dropout(dropout))

    # Flatten layer
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(reg)))

    model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def results_NN(dataset, parameters_list, epochs = 90, cnn=False):
    """
    Train and evaluate ANN or CNN models on given datasets and parameters, plot training/validation F1 scores.

    Parameters:
    dataset (list): List of tuples containing training, validation, and test data.
    parameters_list (list): List of parameter dictionaries for the ANN or CNN models.
    epochs (int): Number of epochs to train each model.
    cnn (bool): Flag to indicate whether to create and train CNN models instead of ANNs.

    Returns:
    list: List of tuples containing train and test accuracy and F1 scores for each dataset.
    """
    results = []

    for idx in range(len(dataset)):

        y_train_dum = dataset[idx][3]
        y_val_dum = dataset[idx][4]
        y_test_dum = dataset[idx][5]
        X_train = dataset[idx][0]
        X_val = dataset[idx][1]
        X_test = dataset[idx][2]

        params = parameters_list[idx]
        if cnn:
            clf = create_CNN(X_train.shape[1],**params)
        else: clf = create_ANN(X_train.shape[1],**params)
        
        clf_hist = clf.fit(X_train, y_train_dum, validation_data=(X_val,y_val_dum), epochs = epochs, verbose=0, batch_size = 32)

        results_data = []

        y_tr_arg=np.argmax(y_train_dum,axis=1)
        Y_pred = np.argmax(clf.predict(X_train),axis=1)
        f1 = f1_score(y_tr_arg,Y_pred, average='weighted')
        acc = accuracy_score(y_tr_arg,Y_pred)

        results_data.append((acc,f1))

        y_test_arg=np.argmax(y_test_dum,axis=1)
        Y_pred = np.argmax(clf.predict(X_test),axis=1)
        f1 = f1_score(y_test_arg,Y_pred, average='weighted')
        acc = accuracy_score(y_test_arg,Y_pred)

        results_data.append((acc,f1))

        results.append(results_data)
        # print("NN Train score with ",format(clf_hist.history["f1_score"][-1]))
        # print("NN Test score with ",format(clf_hist.history["val_f1_score"][-1]))

        plt.figure(figsize=(8, 4))
        plt.plot(clf_hist.history["f1_score"], label="training set")
        plt.plot(clf_hist.history["val_f1_score"], label="validation set")
        plt.xlabel("epochs")
        plt.ylabel("f1_score")
        plt.legend()
        plt.show()

        del clf

    return results

def plot_res(results_list):
    """
    Plot bar graphs comparing training and testing accuracy and F1 scores for various datasets.

    Parameters:
    results_list (list): List of tuples containing accuracy and F1 scores for train and test sets.

    Note:
    Assumes dataset names are ["OHE", "W2V", "D2V", "TF-IDF", "BERT"].
    """
    dataset_names = ["OHE", "W2V", "D2V", "TF-IDF", "BERT"]

    # Plotting Accuracy and F1 Score
    plt.figure(figsize=(12, 6))
    bar_width = 0.5/5
    index = np.arange(len(dataset_names))

    ## TRAIN
    # Extract accuracy, precision, and F1 scores for each dataset
    acc_scores = [result[0][0] for result in results_list]
    f1_scores = [result[0][1] for result in results_list]

    # Barplot for accuracy and F1 scores
    plt.bar(index - bar_width, acc_scores, bar_width, label='Train Accuracy', color='yellow', alpha=0.7)
    plt.bar(index, f1_scores, bar_width, label='Train F1 Score', color='red', alpha=0.7)
    
    ## TEST
    # Extract accuracy, precision, and F1 scores for each dataset
    acc_scores = [result[1][0] for result in results_list]
    f1_scores = [result[1][1] for result in results_list]

    # Barplot for accuracy and F1 scores
    plt.bar(index + bar_width, acc_scores, bar_width, label='Test Accuracy', color='blue', alpha=0.7)
    plt.bar(index + 2*bar_width, f1_scores, bar_width, label='Test F1 Score', color='green', alpha=0.7)

    # Customize the plot
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    #plt.ylim(top=0.5)
    plt.title('Accuracy and F1 Score Comparison')
    plt.xticks(index + bar_width / 2, dataset_names)
    plt.legend()
    plt.grid(axis='y')

    plt.show()

def plot_res_rf(results_list):
    """
    Plot bar graphs comparing training and testing accuracy, F1 scores, and max depth for Random Forest across various datasets.

    Parameters:
    results_list (list): List of tuples containing accuracy, F1 scores, and average max depth for train and test sets.

    Note:
    Assumes dataset names are ["OHE", "W2V", "D2V", "TF-IDF", "BERT"] and includes an additional plot for max depth.
    """
    dataset_names = ["OHE", "W2V", "D2V", "TF-IDF", "BERT"]

    # Plotting Accuracy and F1 Score
    plt.figure(figsize=(12, 6))
    bar_width = 0.5/5
    index = np.arange(len(dataset_names))

    ## TRAIN
    # Extract accuracy, precision, and F1 scores for each dataset
    acc_scores = [result[0][0] for result in results_list]
    f1_scores = [result[0][1] for result in results_list]

    # Barplot for accuracy and F1 scores
    plt.bar(index - bar_width, acc_scores, bar_width, label='Train Accuracy', color='yellow', alpha=0.7)
    plt.bar(index, f1_scores, bar_width, label='Train F1 Score', color='red', alpha=0.7)
    
    ## TEST
    # Extract accuracy, precision, and F1 scores for each dataset
    acc_scores = [result[1][0] for result in results_list]
    f1_scores = [result[1][1] for result in results_list]

    # Barplot for accuracy and F1 scores
    plt.bar(index + bar_width, acc_scores, bar_width, label='Test Accuracy', color='blue', alpha=0.7)
    plt.bar(index + 2*bar_width, f1_scores, bar_width, label='Test F1 Score', color='green', alpha=0.7)

    # Customize the plot
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    #plt.ylim(top=0.5)
    plt.title('Accuracy and F1 Score Comparison')
    plt.xticks(index + bar_width / 2, dataset_names)
    plt.legend()
    plt.grid(axis='y')

    plt.show()

    # Plotting Precision
    plt.figure(figsize=(8, 5))
    bar_width = 0.5/2

    # Barplot for precision
    depth = [result[0][2] for result in results_list]
    plt.bar(index - 1/2*bar_width, depth, bar_width, label='Train Depth', color='orange', alpha=0.7)
    depth = [result[1][2] for result in results_list]
    plt.bar(index + 1/2*bar_width, depth, bar_width, label='Test Depth', color='blue', alpha=0.7)

    # Customize the plot
    plt.xlabel('Dataset')
    plt.ylabel('Max Depth')
    plt.title('Depth Comparison')
    plt.xticks(index, dataset_names)
    plt.legend()

    plt.show()

def results_rf(encoder, dataset, parameters_list):
    """
    Train Random Forest classifiers, compute accuracy and F1 scores, and report average max depth.

    Parameters:
    encoder (LabelEncoder): Encoder instance for transforming labels.
    dataset (list): List of datasets (train and test features and labels).
    parameters_list (list): List of parameter dictionaries for Random Forest models.

    Returns:
    list: Results containing accuracy, F1 score, and average max depth for train and test sets.
    """
    results_list = []

    for idx in range(len(dataset)):
        y_train = encoder.fit_transform(dataset[idx][2])
        y_test = encoder.transform(dataset[idx][3])
        X_train = dataset[idx][0]
        X_test = dataset[idx][1]

        params = parameters_list[idx]

        rf = RandomForestClassifier(**params)
        print(X_train.shape, y_train.shape)
        rf.fit(X_train, y_train)
        print("RF Train score with ",format(rf.score(X_train, y_train)))

        Max_Depth = []
        for tree_idx, tree_estimator in enumerate(rf.estimators_):
            Max_Depth.append(tree_estimator.tree_.max_depth)
        MMD = sum(Max_Depth)/len(Max_Depth)

        results_data = []

        # Train metrics
        y_pred = rf.predict(X_train)
        acc_rf = accuracy_score(y_train, y_pred)
        #precision_rf = precision_score(y_train, y_pred, average='macro')
        #ecall_rf = recall_score(y_train, y_pred, average='macro')
        f1_rf = f1_score(y_train, y_pred, average='macro')
        print("Random Forest Train accuracy {}".format(acc_rf))
        #print("Random Forest Train precision {}".format(precision_rf))
        #print("Random Forest Train recall {}".format(recall_rf))
        print("Random Forest Train f1 {}".format(f1_rf))
        results_data.append((acc_rf, f1_rf, MMD))

        # Test metrics
        y_pred = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred)
        #precision_rf = precision_score(y_test, y_pred, average='macro')
        #recall_rf = recall_score(y_test, y_pred, average='macro')
        f1_rf = f1_score(y_test, y_pred, average='macro')
        print("Random Forest Test accuracy {}".format(acc_rf))
        #print("Random Forest Test precision {}".format(precision_rf))
        #print("Random Forest Test recall {}".format(recall_rf))
        print("Random Forest Test f1 {}".format(f1_rf))
        results_data.append((acc_rf, f1_rf, MMD))

        results_list.append(results_data)
    return results_list

def results_svc(encoder, dataset, parameters_list):
    """
    Train Support Vector Classifier (SVC) models and compute accuracy and F1 scores.

    Parameters:
    encoder (LabelEncoder): Encoder instance for transforming labels.
    dataset (list): List of datasets (train and test features and labels).
    parameters_list (list): List of parameter dictionaries for SVC models.

    Returns:
    tuple: Results containing accuracy and F1 score for train and test sets, and predictions for Breast Cancer cases.
    """
    results_list = []
    predictions_breast_cancer = []

    for idx in range(len(dataset)):
        y_train = encoder.fit_transform(dataset[idx][2])
        y_test = encoder.transform(dataset[idx][3])
        X_train = dataset[idx][0]
        X_test = dataset[idx][1]

        params = parameters_list[idx]

        svc = LinearSVC(**params)
        print(X_train.shape, y_train.shape)
        svc.fit(X_train, y_train)
        print("SVC Train score with ",format(svc.score(X_train, y_train)))

        results_data = []

        # Train metrics
        y_pred = svc.predict(X_train)
        acc_svc = accuracy_score(y_train, y_pred)
        #precision_svc = precision_score(y_train, y_pred, average='macro')
        #ecall_svc = recall_score(y_train, y_pred, average='macro')
        f1_svc = f1_score(y_train, y_pred, average='weighted')
        print("Linear SVC Train accuracy {}".format(acc_svc))
        #print("Linear SVC Train precision {}".format(precision_svc))
        #print("Linear SVC Train recall {}".format(recall_svc))
        print("Linear SVC Train f1 {}".format(f1_svc))
        results_data.append((acc_svc, f1_svc))

        # Test metrics
        y_pred = svc.predict(X_test)
        acc_svc = accuracy_score(y_test, y_pred)
        #precision_svc = precision_score(y_test, y_pred, average='macro')
        #recall_svc = recall_score(y_test, y_pred, average='macro')
        f1_svc = f1_score(y_test, y_pred, average='weighted')
        print("Linear SVC Test accuracy {}".format(acc_svc))
        #print("Linear SVC Test precision {}".format(precision_svc))
        #print("Linear SVC Test recall {}".format(recall_svc))
        print("Linear SVC Test f1 {}".format(f1_svc))
        results_data.append((acc_svc, f1_svc))

        results_list.append(results_data)

        ###### Check Breast Cancer
        predictions = svc.predict(X_test)
        y_test_bc = dataset[idx][3].copy().reset_index(drop=True)
        breast_cancer_indices = y_test_bc[y_test_bc == 'Breast Cancer'].index
        predicted_labels_breast_cancer = predictions[breast_cancer_indices]
        prediction_breast_cancer = encoder.inverse_transform(predicted_labels_breast_cancer)
        print(prediction_breast_cancer)
        predictions_breast_cancer.append(prediction_breast_cancer)
    return results_list, predictions_breast_cancer

def results_gnb(encoder, dataset, parameters_list):
    """
    Train Gaussian Naive Bayes (GNB) models and compute accuracy and F1 scores.

    Parameters:
    encoder (LabelEncoder): Encoder instance for transforming labels.
    dataset (list): List of datasets (train and test features and labels).
    parameters_list (list): List of parameter dictionaries for GNB models.

    Returns:
    list: Results containing accuracy and F1 score for train and test sets.
    """
    results_list = []

    for idx in range(len(dataset)):
        y_train = encoder.fit_transform(dataset[idx][2])
        y_test = encoder.transform(dataset[idx][3])
        X_train = dataset[idx][0]
        X_test = dataset[idx][1]

        params = parameters_list[idx]

        gnb = GaussianNB(**params)
        print(X_train.shape, y_train.shape)
        gnb.fit(X_train, y_train)
        print("gnb Train score with ",format(gnb.score(X_train, y_train)))

        results_data = []

        # Train metrics
        y_pred = gnb.predict(X_train)
        acc_gnb = accuracy_score(y_train, y_pred)
        #precision_gnb = precision_score(y_train, y_pred, average='macro')
        #ecall_gnb = recall_score(y_train, y_pred, average='macro')
        f1_gnb = f1_score(y_train, y_pred, average='weighted')
        print("Gaussian NB Train accuracy {}".format(acc_gnb))
        #print("Gaussian NB Train precision {}".format(precision_gnb))
        #print("Gaussian NB Train recall {}".format(recall_gnb))
        print("Gaussian NB Train f1 {}".format(f1_gnb))
        results_data.append((acc_gnb, f1_gnb))

        # Test metrics
        y_pred = gnb.predict(X_test)
        acc_gnb = accuracy_score(y_test, y_pred)
        #precision_gnb = precision_score(y_test, y_pred, average='macro')
        #recall_gnb = recall_score(y_test, y_pred, average='macro')
        f1_gnb = f1_score(y_test, y_pred, average='weighted')
        print("Gaussian NB Test accuracy {}".format(acc_gnb))
        #print("Gaussian NB Test precision {}".format(precision_gnb))
        #print("Gaussian NB Test recall {}".format(recall_gnb))
        print("Gaussian NB Test f1 {}".format(f1_gnb))
        results_data.append((acc_gnb, f1_gnb))

        results_list.append(results_data)
    return results_list