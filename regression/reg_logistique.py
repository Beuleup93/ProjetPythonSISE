#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 21:09:43 2021

@author: macbookair
"""
#numpy
import numpy as np

#change the current directory
import os
os.chdir("/Users/macbookair/Desktop/Lyon2/Datamining/spam_sms_data")

#importation of the corpus
import pandas
spams = pandas.read_table("SMSSpamCollection.txt",sep="\t",header=0)

#type of the object
print(type(spams))

#size of the dataset
print(spams.shape)

#list of columns
print(spams.columns)

#type of columns
print(spams.dtypes)

#description
print(spams.describe())

#frequency distribution of the class attribute
print(pandas.crosstab(index=spams["classe"],columns="count"))

#**** DECOUPAGE EN TRAIN ET TEST *****

#subdivision into train and test sets
from sklearn.model_selection import train_test_split
spamsTrain, spamsTest = train_test_split(spams,train_size=3572,random_state=1,stratify=spams['classe'])

#frequency distribution of the class attribute
#train set
freqTrain = pandas.crosstab(index=spamsTrain["classe"],columns="count")
print(freqTrain/freqTrain.sum())
#test set
freqTest = pandas.crosstab(index=spamsTest["classe"],columns="count")
print(freqTest/freqTest.sum())

#*** Generate the document term matrix - train set ***

#import the CountVectorizer tool
from sklearn.feature_extraction.text import CountVectorizer

#instantiation of the objet
parseur = CountVectorizer(binary=True)

#create the document term matrix
XTrain = parseur.fit_transform(spamsTrain['message'])

#number of  tokens
print(len(parseur.get_feature_names()))

#list of tokens
print(parseur.get_feature_names())

#transform the sparse matrix into a numpy matrix
mdtTrain = XTrain.toarray()

#type of the matrix
print(type(mdtTrain))

#size of the matrix
print(mdtTrain.shape)

#frequency of the terms
freq_mots = np.sum(mdtTrain,axis=0)
print(freq_mots)

#argmax
index = np.argsort(freq_mots)
print(index)

#print the terms and their frequency
imp = {'terme':np.asarray(parseur.get_feature_names())[index],'freq':freq_mots[index]}
print(pandas.DataFrame(imp))

#*** train the classifier

#import the class LogistiRegression
from sklearn.linear_model import LogisticRegression

#instatiate the object
modelFirst = LogisticRegression()

#perform the training process
modelFirst.fit(mdtTrain,spamsTrain['classe'])

#size of coefficients matrix
print(modelFirst.coef_.shape)

#intercept
print(modelFirst.intercept_)

#generate the document term matrix for the test set
#using the object learned from the train set

#create the document term matrix
mdtTest = parseur.transform(spamsTest['message'])

#size of the matrix
print(mdtTest.shape)

#prediction for the test set
predTest = modelFirst.predict(mdtTest)

#import the metrics class for the performance measurement
from sklearn import metrics

#confusion matrix
mcTest = metrics.confusion_matrix(spamsTest['classe'],predTest)
print(mcTest)

#recall
print(metrics.recall_score(spamsTest['classe'],predTest,pos_label='spam'))

#precision
print(metrics.precision_score(spamsTest['classe'],predTest,pos_label='spam'))

#F1-Score
print(metrics.f1_score(spamsTest['classe'],predTest,pos_label='spam'))

#accuracy rate
print(metrics.accuracy_score(spamsTest['classe'],predTest))

#***** STOP WORDS and MIN FREQUENCY

#rebuild the parser with new options : stop_words='english' and min_df = 10
parseurBis = CountVectorizer(stop_words='english',binary=True, min_df = 10)
XTrainBis = parseurBis.fit_transform(spamsTrain['message'])

#number of tokens
print(len(parseurBis.get_feature_names()))

#mdt_bis
mdtTrainBis = XTrainBis.toarray()

#instatiate the object
modelBis = LogisticRegression()

#perform the training process
modelBis.fit(mdtTrainBis,spamsTrain['classe'])

#generate the document term matrix for the test set
#using the object learned from the train set

#create the document term matrix
mdtTestBis = parseurBis.transform(spamsTest['message'])

#prediction for the test set
predTestBis = modelBis.predict(mdtTestBis)

#confusion matrix
mcTestBis = metrics.confusion_matrix(spamsTest['classe'],predTestBis)
print(mcTestBis)

#recall
print(metrics.recall_score(spamsTest['classe'],predTestBis,pos_label='spam'))

#precision
print(metrics.precision_score(spamsTest['classe'],predTestBis,pos_label='spam'))

#F1-Score
print(metrics.f1_score(spamsTest['classe'],predTestBis,pos_label='spam'))

#accuracy rate
print(metrics.accuracy_score(spamsTest['classe'],predTestBis))

#***** REMOVE TERMS WITH COEFFICIENTS NEARLY ZERO

#absolute  value of the coefficients
coef_abs = np.abs(modelBis.coef_[0,:])

#percentiles of the coefficients (absolute value)
thresholds = np.percentile(coef_abs,[0,25,50,75,90,100])
print(thresholds)

#identify the coefficients "significantly higher than zero
#use 1st quartile as threshold
indices = np.where(coef_abs > thresholds[1])
print(len(indices[0]))

#create the new document term matrices

#document term matrices - train and test sets
mdtTrainTer = mdtTrainBis[:,indices[0]]
mdtTestTer = mdtTestBis[:,indices[0]]

#checking
print(mdtTrainTer.shape)
print(mdtTestTer.shape)

#instatiate the object
modelTer = LogisticRegression()

#train a new classifier with selected terms
modelTer.fit(mdtTrainTer,spamsTrain['classe'])

#prediction on the test set
predTestTer = modelTer.predict(mdtTestTer)

#confusion matrix
mcTestTer = metrics.confusion_matrix(spamsTest['classe'],predTestTer)
print(mcTestTer)

#recall
print(metrics.recall_score(spamsTest['classe'],predTestTer,pos_label='spam'))

#precision
print(metrics.precision_score(spamsTest['classe'],predTestTer,pos_label='spam'))

#F1-Score
print(metrics.f1_score(spamsTest['classe'],predTestTer,pos_label='spam'))

#accuracy rate
print(metrics.accuracy_score(spamsTest['classe'],predTestTer))

#selected terms
sel_terms = np.array(parseurBis.get_feature_names())[indices[0]]

#sorted indices of the absolute value coefficients
sorted_indices = np.argsort(np.abs(modelTer.coef_[0,:]))

#print the terms and theirs coefficients
imp = {'term':np.asarray(sel_terms)[sorted_indices],'coef':modelTer.coef_[0,:][sorted_indices]}
print(pandas.DataFrame(imp))

#deployment

#document to classify
doc = ['this is a new free service for you only']

#get its description
desc = parseurBis.transform(doc)
print(desc)

#which terms
print(np.asarray(parseurBis.get_feature_names())[desc.indices])

#dense representation
dense_desc = desc.toarray()

#apply var. selection
dense_sel = dense_desc[:,indices[0]]

#prediction of the class membership
pred_doc = modelTer.predict(dense_sel)
print(pred_doc)

#prediction of the class membership probabilities
pred_proba = modelTer.predict_proba(dense_sel)
print(pred_proba)

#checking - logit
logit = 1.823208 + 1.999089 + 2.201104 + modelTer.intercept_

#probability - logistic function
import math
p_spam = 1/(1+math.exp(-logit))
print(p_spam)


