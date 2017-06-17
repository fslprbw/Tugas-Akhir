import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import csv
from py4j.java_gateway import JavaGateway
import re
from nltk.util import ngrams
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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

		list_of_label.append(label)
		list_of_comment.append(comment)


#10 fold CV
cv = CountVectorizer()
num_folds = 10
size = int(len(list_of_comment[:90]))
subset_size = int(len(list_of_comment[:90])/num_folds)
sum_NB_acc = 0
sum_DT_acc = 0
sum_SVM_acc = 0
result_NB_label = []
result_DT_label = []
result_SVM_label = []

X = cv.fit_transform(list_of_comment).toarray()
Y = np.array(list_of_label)

X = X[:90]
Y = Y[:90]

# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=2)
# for train, test in skf.split(X,Y):
# 	print("%s %s" % (train, test))

# for i in range(0, len(X), 8):
# 	test = X[i:i+8]
# 	train = []
# 	for j in range(0, len(X), 8):
# 		if i != j: train.append(X[i:i+8])
# print "test", test
# print "train", train

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

	clf = GaussianNB()
	GaussianNB(priors=None)
	clf.fit(train_comment_round, train_label_round)
	result_NB_label = np.concatenate((result_NB_label,clf.predict(test_comment_round)))
	sum_NB_acc += metrics.accuracy_score(test_label_round, clf.predict(test_comment_round))


	clf = DecisionTreeClassifier()
	clf.fit(train_comment_round, train_label_round)
	result_DT_label = np.concatenate((result_DT_label,clf.predict(test_comment_round)))
	sum_DT_acc += metrics.accuracy_score(test_label_round, clf.predict(test_comment_round))

	clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
	clf.fit(train_comment_round, train_label_round)
	result_SVM_label = np.concatenate((result_SVM_label,clf.predict(test_comment_round)))
	sum_SVM_acc += metrics.accuracy_score(test_label_round, clf.predict(test_comment_round))
	
print "-- Naive Bayes --"
print confusion_matrix(Y,result_NB_label, labels=["dijawab", "dibaca", "dihitung", "diabaikan"])
print sum_NB_acc/num_folds

print "-- Decision Tree --"
print confusion_matrix(Y,result_DT_label, labels=["dijawab", "dibaca", "dihitung", "diabaikan"])
print sum_DT_acc/num_folds

print "-- Support Vector Machine --"
print confusion_matrix(Y,result_SVM_label, labels=["dijawab", "dibaca", "dihitung", "diabaikan"])
print sum_SVM_acc/num_folds