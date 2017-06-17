# -*- coding: unicode-escape -*-
import os, sys

import nltk
import csv
from py4j.java_gateway import JavaGateway
import re
import numpy as np
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def pre_process(text):
	#Convert to lower case
	text = ''.join(text).lower()
	# Convert www.* or https?://* to URL
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' _URL_ ',text)
	#Convert @username to AT_USER
	text = re.sub('@'+poster,' _mentionpemilik_ ',text)
	text = re.sub('@[^\s]+',' _mentionteman_ ',text)	
	#Replace #word with word
	text = re.sub(r'#([^\s]+)', r'\1', text)
	#Remove koma
	text = re.sub('[,]+', '', text)
	#Remove koma
	text = re.sub('[.]+', ' _tanda_titik_ ', text)
	#Remove koma
	text = re.sub('[?]+', ' _tanda_tanya_ ', text)
	#Remove koma
	text = re.sub('[!]+', ' _tanda_seru_ ', text)
	#Remove additional white spaces
	text = re.sub('[\s]+', ' ', text)
	return text

# def feature_extraction(text, sentence_number, label):
# 	feature_of_sentence = []
# 	c = Counter(text)
# 	text = list(c)
# 	for i in range(len(text)):
# 		feature = [] #(feature, count, kalimat, label)
# 		feature.append(text[i])
# 		feature.append(c[text[i]])
# 		feature.append(sentence_number)
# 		feature.append(label)
# 		feature_of_sentence.append(feature)
# 	return feature_of_sentence

def feature_extraction(text):
	# feature = {}
	c = Counter(text)
	text = list(c)
	return text
	# for i in range(len(text)):
	# 	feature[text[i]] = "true"
	# return feature

def read_csv(file_directory):
	instagram_data = []
	with open(file_directory) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data = []
			data.append(row['id'])
			data.append(row['content'])
			data.append(row['main post'])
			data.append(row['label'])
			instagram_data.append(data)
	return instagram_data

def show_feature_info(bag_of_feature):
	count = Counter(bag_of_feature)
	print 'Jumlah Feature = ', len(list(count))
	print "== All Features =="
	print count

#algorthm --> NB for naive bayes, DT for decision tree, SVM for support vector machie
def cross_fold_validation(number_of_fold, number_of_data_tested, list_of_comment, list_of_label, algorithm):
	cv = CountVectorizer()
	num_folds = number_of_fold
	size = number_of_data_tested
	subset_size = number_of_data_tested/num_folds
	sum_NB_acc = 0
	sum_DT_acc = 0
	sum_SVM_acc = 0
	result_NB_label = []
	result_DT_label = []
	result_SVM_label = []

	X = cv.fit_transform(list_of_comment).toarray()
	Y = np.array(list_of_label)

	X = X[:number_of_data_tested]
	Y = Y[:number_of_data_tested]

	for index in range (0, size, subset_size):
		test_comment_round = X[index:index+subset_size]
		test_label_round = Y[index:index+subset_size]
		train_comment_round = []
		train_label_round = []
		# print "train"
		# print train_label_round
		for j in range(0, size, subset_size):
			if index != j:
				if (len(train_comment_round) == 0):
					train_comment_round = (X[j:j+subset_size])
					train_label_round = (Y[j:j+subset_size])
				else:
					train_comment_round = np.concatenate((train_comment_round,X[j:j+subset_size]))
					train_label_round = np.concatenate((train_label_round,Y[j:j+subset_size]))

		if algorithm == "NB":
			clf = GaussianNB()
			GaussianNB(priors=None)
			clf.fit(train_comment_round, train_label_round)
			result_NB_label = np.concatenate((result_NB_label,clf.predict(test_comment_round)))
			sum_NB_acc += metrics.accuracy_score(test_label_round, clf.predict(test_comment_round))

		elif algorithm == "DT":

			clf = DecisionTreeClassifier()
			clf.fit(train_comment_round, train_label_round)
			result_DT_label = np.concatenate((result_DT_label,clf.predict(test_comment_round)))
			sum_DT_acc += metrics.accuracy_score(test_label_round, clf.predict(test_comment_round))

		elif algorithm == "SVM":

			clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
		    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
		    max_iter=-1, probability=False, random_state=None, shrinking=True,
		    tol=0.001, verbose=False)
			clf.fit(train_comment_round, train_label_round)
			result_SVM_label = np.concatenate((result_SVM_label,clf.predict(test_comment_round)))
			sum_SVM_acc += metrics.accuracy_score(test_label_round, clf.predict(test_comment_round))
		
	if algorithm == "NB":
		print "-- Naive Bayes --"
		print confusion_matrix(Y,result_NB_label, labels=["dijawab", "dibaca", "dihitung", "diabaikan"])
		print sum_NB_acc/num_folds

	elif algorithm == "DT":
		print "-- Decision Tree --"
		print confusion_matrix(Y,result_DT_label, labels=["dijawab", "dibaca", "dihitung", "diabaikan"])
		print sum_DT_acc/num_folds

	elif algorithm == "SVM":
		print "-- Support Vector Machine --"
		print confusion_matrix(Y,result_SVM_label, labels=["dijawab", "dibaca", "dihitung", "diabaikan"])
		print sum_SVM_acc/num_folds

	else :
		 	print " Algorithm not Found"


list_of_data = []
# list_of_feature = []
list_of_label = []
list_of_comment = []
processed_data = []
bag_of_feature  = []

list_of_data = read_csv('../Resource/sample_labelled.csv')

for index in range(len(list_of_data)):
	poster_status = list_of_data[index][2]
	if poster_status == 'yes':
		poster = list_of_data[index][0]
	else:
		comment = list_of_data[index][1]
		label = list_of_data[index][3]

		processed_comment = pre_process(comment)
		# processed_comment = nltk.word_tokenize(comment)
		print feature_extraction(nltk.word_tokenize(processed_comment))
		bag_of_feature += feature_extraction(nltk.word_tokenize(processed_comment))
		# list_of_feature.append(feature_extraction(processed_comment))
		list_of_label.append(label)
		list_of_comment.append(processed_comment)

show_feature_info(bag_of_feature)

cross_fold_validation(10, 90, list_of_comment, list_of_label, "SVM")
cross_fold_validation(10, 90, list_of_comment, list_of_label, "DT")
cross_fold_validation(10, 90, list_of_comment, list_of_label, "NB")
