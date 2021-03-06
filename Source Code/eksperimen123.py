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
from sklearn.externals import joblib

def pre_process(text):
	#turn emoticon to unicode
	text = unicode(text, 'utf-8')
	text = text.encode('unicode_escape')
	#convert unicode of newline to newline
	text = re.sub(r'\\n',' ',text)
	# # Convert www.* or https?://* to URL
	# text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' _url_ ',text)
	#Replace #word with word
	text = re.sub(r'#([^\s]+)',' _hashtag_ ', text)
	# Convert @username to AT_USER
	text = re.sub('@'+poster,' _mentionpemilik_ ',text)
	text = re.sub('@[^\s]+',' _mentionteman_ ',text)	
	#Convert mark
	text = re.sub('[/]+', ' ', text)
	text = re.sub('[&]+', ' ', text)
	text = re.sub('[,]+', ' ', text)
	text = re.sub('[.]+', ' _tandatitik_ ', text)
	text = re.sub('[?]+', ' _tandatanya_ ', text)
	text = re.sub('[!]+', ' _tandaseru_ ', text)
	#convert emoticon and symbol
	text = re.sub(r'\\U000[^\s]{5}',convert_emoticon,text)
	# text = re.sub(r'\\u[\d][^\s]{3}',' _emoticon_ ',text)
	#convert digit
	text = re.sub('[\d]+', ' _angka_ ', text)
	# Remove additional white spaces
	text = re.sub('[\s]+', ' ', text)
	#Convert to lower case
	text = ''.join(text).lower()
	return text

def convert_emoticon (text):

	emot_positif = ["u0001f600","u0001f601","u0001f602","u0001f923","u0001f603","u0001f604","u0001f605","u0001f606","u0001f607","u0001f609","u0001f60a","u0001f60b","u0001f60e","u0001f60d","u0001f60e","u0001f618","u0001f617","u0001f618","u0001f619","u0001f61a","u0001f63a","u0001f642","u0001f917","u0001f929","u0001f44a","u0001f44c","u0001f44d","u0001f44f","u0001f495","u0001f496","u0001f49c","u0001f49e",":)",":-)",":D",":-D",":*"]
	emot_negatif = ["u0001f608","u0001f641","u0001f616","u0001f61e","u0001f61f","u0001f624","u0001f622","u0001f62d","u0001f626","u0001f627","u0001f628","u0001f629","u0001f92f","u0001f62c","u0001f630","u0001f631","u0001f633","u0001f92a","u0001f635","u0001f63f","u0001f621","u0001f620","u0001f92c","u0001f494",":(",":-(",";(",";-("]
	emot_netral = ["u0001f914","u0001f928","u0001f610","u0001f611","u0001f612","u0001f614","u0001f636","u0001f644","u0001f64b","u0001f64c","u0001f64f","u0001f68c","u0001f60f","u0001f623","u0001f625","u0001f62e","u0001f910","u0001f62f","u0001f62a","u0001f62b","u0001f634","u0001f60c","u0001f61b","u0001f61c","u0001f61d","u0001f924","u0001f612","u0001f613","u0001f614","u0001f615","u0001f643","u0001f911","u0001f632","u0001f1f0","u0001f1f5","u0001f334","u0001f338","u0001f34e","u0001f3b6","u0001f3ba","u0001f3fb","u0001f3fc","u0001f479","u0001f47b","u0001f483","u0001f48b","u0001f4e2","u0001f4e3"]
	result = ""

	if text.group(0).lower()[1:] in emot_positif:
		result = " _emotpos_ "
	elif text.group(0).lower()[1:] in emot_negatif:
		result = " _emotneg_ "
	else:
		result = " _emotnetral_ "

	return result

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
	print "Seleksi FItur = ", number_of_feature
	new_X = X

	list_of_gain = []
	Z = information_gain(X,Y)
	for index in range(len(Z)):
		feature_gain = [Z[index],index]
		list_of_gain.append(feature_gain)

	list_of_gain = sorted(list_of_gain, reverse=True)

	sorted_feature = []

	cv_temp = joblib.load('countvec.pkl')
	for sort in range(len(list_of_gain)):
		test = "" 
		test += cv_temp.get_feature_names()[int(list_of_gain[sort][1])]
		print test
		sorted_feature.append(test)

	printToCSV(sorted_feature, "sorted feature selection")

	selected_feature = []

	if (number_of_feature > len(Z)):
		print "Nilai seleksi fitur melebihi jumlah feature", number_of_feature, ":", len(Z)
		number_of_feature = len(Z)

	for index in range(number_of_feature):
		selected_feature.append(list_of_gain[index][1])

	for index in range(len(X[0])):
		if (index not in selected_feature):
			for a in range(len(X)):
				new_X[a][index] = 0
		# else:
		# 	cv_temp = joblib.load('countvec.pkl')
		# 	print cv_temp.get_feature_names()[index]

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
	# cv = CountVectorizer(input='content', binary=True, tokenizer=lambda text:nltk.word_tokenize(text))

	X = cv.fit_transform(list_of_comment).toarray()
	Y = np.array(list_of_label)

	joblib.dump(cv, 'countvec.pkl')

	X = feature_selection(X,Y, fitur)

	print "Total Kata = ", len(X[0])

	# printToCSV(cv.get_feature_names(), "kata")

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

		# train_comment_round = feature_selection(train_comment_round,train_label_round, fitur)

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


def classify(train_comment, train_label, test_comment, algorithm):

	cv = CountVectorizer()
	# cv = CountVectorizer(input='content', binary=True, tokenizer=lambda text:nltk.word_tokenize(text))

	X = cv.fit_transform(list_of_comment).toarray()
	Y = np.array(list_of_label)

	word_list = cv.get_feature_names()
	joblib.dump(word_list, 'unigram_wordlist.pkl')

	comment_vector = np.zeros((1, len(word_list)))

	word = nltk.word_tokenize(test_comment)

	for index in range(len(word_list)):
		if word_list[index] in word:
			comment_vector[0][index] += 1

	if algorithm == "NB":
		clf = GaussianNB()
		GaussianNB(priors=None)
		clf.fit(X, Y)
		joblib.dump(clf, 'unigram-NB_model.pkl')
		print clf.predict(comment_vector)[0]

	elif algorithm == "DT":

		clf = DecisionTreeClassifier()
		clf.fit(X, Y)
		joblib.dump(clf, 'unigram-DT_model.pkl')
		print clf.predict(comment_vector)[0]

	elif algorithm == "SVM":

		# X = feature_selection(X,Y, 250)

		clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
	    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
	    max_iter=-1, probability=False, random_state=None, shrinking=True,
	    tol=0.001, verbose=False)
		clf.fit(X, Y)
		joblib.dump(clf, 'baseline-SVM_model.pkl')
		print clf.predict(comment_vector)[0]
	
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
			# inlp_output = remove_stopword(inlp_output)

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
			# inlp_output = remove_stopword(inlp_output)

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

			# processed_comment = pre_process(comment)
			list_of_label.append(label)
			# list_of_comment.append(processed_comment)
			list_of_comment.append(comment)

# list_of_comment = inlppreproses(list_of_comment)

printToCSV(list_of_comment, "komen dengan preproses")
# printToCSV(list_of_label, "label awal")

print "Jumlah data awal :", len(list_of_data)
print "Jumlah data model :", len(list_of_label)
data_distribution(list_of_label)

# test_sentence = "alhamdulillah _tandatitik_ terima kasih _emot_pos_ _emot_pos_"

for repeat in range(1):

	# start = time.time()
	# classify(list_of_comment, list_of_label, test_sentence, "DT")
	# end = time.time()
	# print "Waktu = ", end-start

	# start = time.time()
	# classify(list_of_comment, list_of_label, test_sentence, "NB")
	# end = time.time()
	# print "Waktu = ", end-start

	# start = time.time()
	# classify(list_of_comment, list_of_label, test_sentence, "SVM")
	# end = time.time()
	# print "Waktu = ", end-start

	# start = time.time()
	# cross_fold_validation(10, list_of_comment, list_of_label, "SVM", 750)
	# end = time.time()
	# print "Waktu = ", end-start

	start = time.time()
	cross_fold_validation(10, list_of_comment, list_of_label, "SVM", 250)
	# print "Fitur dipilih = ", 1250+(repeat*250)
	end = time.time()
	print "Waktu = ", end-start