# -*- coding: unicode-escape -*-
import os, sys

import nltk
import csv
from py4j.java_gateway import JavaGateway
import re
from nltk.util import ngrams
from collections import Counter

def pre_process(text):
	#Convert to lower case
	text = ''.join(text).lower()
	# Convert www.* or https?://* to URL
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','_URL_',text)
	#Convert @username to AT_USER
	text = re.sub('@'+poster,'_mentionpemilik_',text)
	text = re.sub('@[^\s]+','_mentionteman_',text)
	#Remove additional white spaces
	text = re.sub('[\s]+', ' ', text)
	#Replace #word with word
	text = re.sub(r'#([^\s]+)', r'\1', text)
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
	feature = {}
	c = Counter(text)
	text = list(c)
	for i in range(len(text)):
		feature[text[i]] = "true"
	return feature

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

list_of_data = []
list_of_feature = []
list_of_label = []
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

		# processed_comment = nltk.word_tokenize(pre_process(comment))
		processed_comment = nltk.word_tokenize(comment)
		list_of_feature.append(feature_extraction(processed_comment))
		list_of_label.append(label)

feature_sets = [(list_of_feature[i],list_of_label[i]) for i in range(len(list_of_feature))]
train_set = feature_sets[50:]
test_set = feature_sets[:50]
naive_bayes_classifier = nltk.NaiveBayesClassifier.train(train_set)
decision_tree_clasifier = nltk.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)

print "naive bayes accuracy = "+ nltk.classify.accuracy(naive_bayes_classifier, test_set)
print "decision tree accuracy = "+ nltk.classify.accuracy(decision_tree_clasifier, test_set)


classifier.show_most_informative_features(5)

# count = Counter(bag_of_feature)
# print count
# print 'Jumlah Feature = ', len(list(count))
# print (list_of_feature)

# for i in range(len(list_of_feature)):
# 	if list_of_feature[i][0] == 'bobow':
# 		print(list_of_feature[i][2], list_of_feature[i][1], list_of_feature[i][3])
