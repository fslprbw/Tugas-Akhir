# -*- coding: unicode-escape -*-
import os, sys

import nltk
import time
import csv
from py4j.java_gateway import JavaGateway
import re
import numpy as np
import scipy as scipy
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from subprocess import Popen, PIPE, STDOUT
from sklearn.feature_selection import VarianceThreshold

def pre_process(text):
	#turn emoticon to unicode
	text = unicode(text, 'utf-8')
	text = text.encode('unicode_escape')
	#convert unicode of newline to newline
	text = re.sub(r'\\n',' ',text)
	# Convert www.* or https?://* to URL
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' _URL_ ',text)
	#Replace #word with word
	text = re.sub(r'#([^\s]+)',' _hashtag_ ', text)
	Convert @username to AT_USER
	text = re.sub('@'+poster,' _mentionpemilik_ ',text)
	text = re.sub('@[^\s]+',' _mentionteman_ ',text)	
	#Convert mark
	text = re.sub('[,]+', ' ', text)
	text = re.sub('[.]+', ' _tanda_titik_ ', text)
	text = re.sub('[?]+', ' _tanda_tanya_ ', text)
	text = re.sub('[!]+', ' _tanda_seru_ ', text)
	#convert emoticon and symbol
	text = re.sub(r'\\U000[^\s]{5}',' _emoticon_ ',text)
	Remove additional white spaces
	text = re.sub('[\s]+', ' ', text)
	#Convert to lower case
	text = ''.join(text).lower()
	return text

def formalization (text):
	result = ""
	p = Popen(['java', '-jar', 'INLPPreproses.jar', 'formalization', text], stdout=PIPE, stderr=STDOUT)
	result = ""
	for line in p.stdout:
		result = line
	return result

def remove_stopword (text):
	result = ""
	p = Popen(['java', '-jar', 'INLPPreproses.jar', 'remove_stopword', text], stdout=PIPE, stderr=STDOUT)
	result = ""
	for line in p.stdout:
		result = line
	return result

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
	# print "== All Features =="
	# print count

def data_distribution(list_of_label):
	jawab = 0
	baca = 0
	abaikan = 0
	for index in range(len(list_of_label)):
		if list_of_label[index] == "jawab":
			jawab += 1
		elif list_of_label[index] == "baca":
			baca += 1			
		elif list_of_label[index] == "abaikan":
			abaikan += 1
		else:
			print index
			print list_of_comment[index-1]
	print "jawab : ", jawab, " - baca : ", baca, " - abaikan : ", abaikan

def information_gain(X, y):

    def _calIg():
        entropy_x_set = 0
        entropy_x_not_set = 0
        for c in classCnt:
            probs = classCnt[c] / float(featureTot)
            entropy_x_set = entropy_x_set - probs * np.log(probs)
            probs = (classTotCnt[c] - classCnt[c]) / float(tot - featureTot)
            entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        for c in classTotCnt:
            if c not in classCnt:
                probs = classTotCnt[c] / float(tot - featureTot)
                entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                             +  ((tot - featureTot) / float(tot)) * entropy_x_not_set)

    tot = X.shape[0]
    classTotCnt = {}
    entropy_before = 0
    for i in y:
        if i not in classTotCnt:
            classTotCnt[i] = 1
        else:
            classTotCnt[i] = classTotCnt[i] + 1
    for c in classTotCnt:
        probs = classTotCnt[c] / float(tot)
        entropy_before = entropy_before - probs * np.log(probs)

    nz = X.T.nonzero()
    pre = 0
    classCnt = {}
    featureTot = 0
    information_gain = []
    for i in range(0, len(nz[0])):
        if (i != 0 and nz[0][i] != pre):
            for notappear in range(pre+1, nz[0][i]):
                information_gain.append(0)
            ig = _calIg()
            information_gain.append(ig)
            pre = nz[0][i]
            classCnt = {}
            featureTot = 0
        featureTot = featureTot + 1
        yclass = y[nz[1][i]]
        if yclass not in classCnt:
            classCnt[yclass] = 1
        else:
            classCnt[yclass] = classCnt[yclass] + 1
    ig = _calIg()
    information_gain.append(ig)

    return np.asarray(information_gain)


def feature_selection (X,Y, number_of_feature):
	new_X = X

	Z = information_gain(X,Y)
	print "Jumlah Kata = ", len(Z)
	Z2 = sorted(Z, reverse=True)

	for index in range(len(Z)):
		if (Z[index] <= Z2[number_of_feature]):
			for a in range(len(X)):
				new_X[a][index] = 0

	return new_X

def print_wrong_class (comment, assigned_label, result_label):
	for index in range(len(assigned_label)):
		if assigned_label[index] != result_label[index]:
			print comment[index]

def printToCSV (data_list, filename):
	with open('../Resource/'+filename+'.csv', 'w') as csvfile:
	    fieldnames = ['no', 'word']
	    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
	    writer.writeheader()

	    for index in range(len(data_list)):
	    	writer.writerow({'no':index, 'word':data_list[index]})

#algorthm --> NB for naive bayes, DT for decision tree, SVM for support vector machie
def cross_fold_validation(number_of_fold, list_of_comment, list_of_label, algorithm,fitur):
	num_folds = number_of_fold
	size = len(list_of_label)
	subset_size = size/num_folds
	sum_NB_acc = 0
	sum_DT_acc = 0
	sum_SVM_acc = 0
	result_NB_label = []
	result_DT_label = []
	result_SVM_label = []
	kata = []

	cv = CountVectorizer()

	X = cv.fit_transform(list_of_comment).toarray()
	Y = np.array(list_of_label)

	# X = feature_selection(X,Y, fitur)

	print "Total Kata = ", len(X[0])

	printToCSV(cv.get_feature_names(), "kata")

	for index in range(10):
		if index < size % num_folds:
			test_start = index*(subset_size+1)
			test_finish = test_start + subset_size + 1
		else:
			test_start = (index*subset_size) + (size % num_folds)
			test_finish = test_start + subset_size
		
		test_comment_round = X[test_start:test_finish]
		test_label_round = Y[test_start:test_finish]
		train_comment_round = []
		train_label_round = []
		for j in range(10):
			if j < size % num_folds:
				train_start = j*(subset_size+1)
				train_finish = train_start + subset_size + 1
			else:
				train_start = (j*subset_size) + (size % num_folds)
				train_finish = train_start + subset_size
			if index != j:
				if (len(train_comment_round) == 0):
					train_comment_round = (X[train_start:train_finish])
					train_label_round = (Y[train_start:train_finish])
				else:
					train_comment_round = np.concatenate((train_comment_round,X[train_start:train_finish]))
					train_label_round = np.concatenate((train_label_round,Y[train_start:train_finish]))

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
		print confusion_matrix(Y,result_NB_label, labels=["jawab", "baca", "abaikan"])
		print sum_NB_acc/num_folds
		print (classification_report(Y, result_NB_label, target_names=["jawab", "baca", "abaikan"]))
		printToCSV(result_NB_label, "hasil_NB")

	elif algorithm == "DT":
		print "-- Decision Tree --"
		print confusion_matrix(Y,result_DT_label, labels=["jawab", "baca", "abaikan"])
		print sum_DT_acc/num_folds
		print (classification_report(Y, result_DT_label, target_names=["jawab", "baca", "abaikan"]))
		printToCSV(result_DT_label, "hasil_DT")

	elif algorithm == "SVM":
		print "-- Support Vector Machine --"
		print confusion_matrix(Y,result_SVM_label, labels=["jawab", "baca", "abaikan"])
		print sum_SVM_acc/num_folds
		print (classification_report(Y, result_SVM_label, target_names=["jawab", "baca", "abaikan"]))
		printToCSV(result_SVM_label, "hasil_SVM")

	else :
		 	print " Algorithm not Found"


def inlppreproses (list_of_comment):
	list_of_processed_comment = []
	index_start = 0
	index_finish = 0
	packet = 1000
	div_value = len(list_of_comment)//packet
	mod_value = len(list_of_comment)%packet

	for repeat in range(div_value+1):
		inlp_input = ""
		inlp_output = ""

		if (repeat < div_value):
			for index in range(packet):
				inlp_input += list_of_comment[repeat*packet + index]
				inlp_input += "`"

			inlp_output = formalization(inlp_input)
			inlp_output = remove_stopword(inlp_output)

			for index in range(len(inlp_output)):
				if (inlp_output[index] == "`"):
					index_finish = index
					list_of_processed_comment.append(inlp_output[index_start:index_finish])
					index_start = index + 1
		else:
			for index in range(mod_value):
				inlp_input += list_of_comment[repeat*packet + index]
				inlp_input += "`"

			inlp_output = formalization(inlp_input)
			inlp_output = remove_stopword(inlp_output)

			for index in range(len(inlp_output)):
				if (inlp_output[index] == "`"):
					index_finish = index
					list_of_processed_comment.append(inlp_output[index_start:index_finish])
					index_start = index + 1

	return list_of_processed_comment

list_of_data = []
# list_of_feature = []
list_of_label = []
list_of_comment = []
processed_data = []

list_of_data = read_csv('../Resource/all_labeled.csv')

for index in range(len(list_of_data)):
	poster_status = list_of_data[index][2]
	if poster_status == 'yes':
		poster = list_of_data[index][0]
	else:
		if list_of_data[index][0] != poster:
			comment = list_of_data[index][1]
			label = list_of_data[index][3]

			processed_comment = pre_process(comment)
			list_of_label.append(label)
			list_of_comment.append(processed_comment)

# list_of_comment = inlppreproses(list_of_comment)

print "Jumlah data awal :", len(list_of_data)
print "Jumlah data model :", len(list_of_label)
data_distribution(list_of_label)

start = time.time()
cross_fold_validation(10, list_of_comment, list_of_label, "DT", 4294)
end = time.time()
print "Waktu = ", end-start

start = time.time()
cross_fold_validation(10, list_of_comment, list_of_label, "NB", 4294)
end = time.time()
print "Waktu = ", end-start

start = time.time()
cross_fold_validation(10, list_of_comment, list_of_label, "SVM", 4294)
end = time.time()
print "Waktu = ", end-start
