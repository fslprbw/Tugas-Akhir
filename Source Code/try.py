import nltk
import csv
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering


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

def pre_process(text):
	# #turn emoticon to unicode
	# text = unicode(text, 'utf-8')
	# text = text.encode('unicode_escape')
	# #convert emoticon
	# text = re.sub(r'\\U000[^\s]{5}',' _emoticon_ ',text)
	# #convert unicode of newline to newline
	# text = re.sub(r'\\n','',text)
	# #Convert to lower case
	# text = ''.join(text).lower()
	# # Convert www.* or https?://* to URL
	# text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' _URL_ ',text)
	# #Convert @username to AT_USER
	# text = re.sub('@'+poster,' _mentionpemilik_ ',text)
	# text = re.sub('@[^\s]+',' _mentionteman_ ',text)	
	# #Replace #word with word
	# text = re.sub(r'#([^\s]+)', r'\1', text)
	# #Remove koma
	# text = re.sub('[,]+', '', text)
	# #Remove koma
	# text = re.sub('[.]+', ' _tanda_titik_ ', text)
	# #Remove koma
	# text = re.sub('[?]+', ' _tanda_tanya_ ', text)
	# #Remove koma
	# text = re.sub('[!]+', ' _tanda_seru_ ', text)
	# #Remove additional white spaces
	# text = re.sub('[\s]+', ' ', text)
	return text

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
		if list_of_data[index][0] != poster:
			comment = list_of_data[index][1]
			label = list_of_data[index][3]

			processed_comment = pre_process(comment)
			# processed_comment = nltk.word_tokenize(comment)
			# print feature_extraction(nltk.word_tokenize(processed_comment))
			# bag_of_feature += feature_extraction(nltk.word_tokenize(comment))
			# list_of_feature.append(feature_extraction(processed_comment))
			list_of_label.append(label)
			list_of_comment.append(processed_comment)

cv = CountVectorizer()
X = cv.fit_transform(list_of_comment).toarray()
# new_data = []
# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# kmeans = KMeans(n_clusters=4).fit(X)
kmeans = AgglomerativeClustering(n_clusters=4).fit(X)
# kmeans = SpectralClustering(n_clusters=4).fit(X)
print len(kmeans.labels_), "--", len(list_of_label)
# for index in range(len(kmeans.labels_)):
# 	if kmeans.labels_[index] == 0:
# 		if (len(new_data) == 0):
# 			new_data = (X[index:index+1])
# 		else:
# 			new_data = np.concatenate((new_data, X[index:index+1]))
			
kmeans = AgglomerativeClustering(n_clusters=2).fit(X)
a = 0
b = 0
c = 0
d = 0
abai = 0
jawab = 0
baca = 0
hitung = 0
for index in range(len(kmeans.labels_)):
	if kmeans.labels_[index] == 0:
		a += 1
		if list_of_label[index] == "jawab":
			jawab += 1
		elif list_of_label[index] == "baca":
			baca += 1
		elif list_of_label[index] == "hitung":
			hitung += 1
		elif list_of_label[index] == "abaikan":
			abai += 1
	elif kmeans.labels_[index] == 1:
		b += 1
		
	elif kmeans.labels_[index] == 2:
		c += 1
		
	elif kmeans.labels_[index] == 3:
		d += 1

		
print jawab, baca, hitung, abai
print a, b, c, d






# print kmeans.labels_
# print list_of_label
# print confusion_matrix(list_of_label, kmeans.labels_)
# print kmeans.cluster_centers_
