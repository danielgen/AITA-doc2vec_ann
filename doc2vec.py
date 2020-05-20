# -*- coding: utf-8 -*-
"""
Created on Wed May 20 00:02:32 2020

@author: genti
"""
import numpy as np
import pandas as pd
from analysis import clean
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

# importing /aita data, concat title and body of post, drop nulls, prune edits
df = pd.read_csv("aita_clean.csv")
df["text"] = df["title"] + " " + df["body"]
df.dropna(subset=["text"], inplace=True)
df["text"] = df["text"].apply(lambda t: t.lower().split("**edit**")[0])

# clean posts words with re, rm stopwords and .pkl vars, get labels
posts = [clean(post, stem=False) for post in tqdm(list(df["text"]))]
labels = list(df["is_asshole"])

# Represent posts as tagged documents and train model
posts = [TaggedDocument(post, [i]) for i, post in enumerate(posts)]
model = Doc2Vec(posts, vector_size=200, window=8, min_count=2, workers=4)

# # loading pretrained model
# model = Doc2Vec.load("aita_doc2vec")

# get post vectors and labels
X = np.array([model.docvecs[i] for i in range(len(posts))])
y = np.array(labels)

# importing ML libraries
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, GridSearchCV

# split in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
# creating model template
def build_clf(optimiser):
    # ann model
    clf = Sequential()
    
    # input and 1st hidden layer (neurons are input size/2), ReLU = 0 v 1
    clf.add(Dense(units=100, 
                  kernel_initializer="uniform", 
                  activation="relu", 
                  input_dim=200))
    
    # dropout between 1st and 2nd layer (random switch-off 30% outputs)
    clf.add(Dropout(0.3))
    
    # second layer (same number of neurons as 1st layer), ReLU = 0 v 1
    clf.add(Dense(units=100, 
                  kernel_initializer="uniform", 
                  activation="relu"))
    
    # dropout towards output node (random switch-off 30% outputs)
    clf.add(Dropout(0.3))
    
    # one output neuron, sigmoid = 0...1
    clf.add(Dense(units=1, 
                  kernel_initializer="uniform", 
                  activation="sigmoid"))
    
    # binary label, interested in accuracy of classification
    clf.compile(optimizer=optimiser, 
                loss="binary_crossentropy", 
                metrics=["accuracy"])
    return clf

# initializing KerasClassifier
classifier = KerasClassifier(build_fn=build_clf)

# defining parameters to tune the model
params = {"batch_size": [32, 128],
          "epochs": [300, 600],
          "optimiser": ["adam", "adadelta"]}

# defining grid search using parameters on classifier, 7 convolutions
gs = GridSearchCV(estimator=classifier, 
                  param_grid=params,
                  scoring="accuracy",
                  cv=7)

# fitting grid search on training data
gs = gs.fit(X_train, y_train)

# getting outcome from grid search
selected_parameters = gs.best_params_
acc = gs.best_score_
winning_clf = gs.best_estimator_

# predictinig on test set
y_pred = winning_clf.predict(X_test) 
y_pred = y_pred > 0.5

# computing metrics on test set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# confusion matrix, classification report and accuracy score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

cr = classification_report(y_test, y_pred)
print ("Classification report:\n", cr)

result2 = accuracy_score(y_test,y_pred)
print("Accuracy:\n", result2)