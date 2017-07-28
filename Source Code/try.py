import nltk
import csv
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from collections import Counter
from nltk.util import ngrams
from math import log10
from decimal import *
from subprocess import Popen, PIPE, STDOUT


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
	#turn emoticon to unicode
	text = unicode(text, 'utf-8')
	text = text.encode('unicode_escape')
	#convert emoticon
	text = re.sub(r'\\U000[^\s]{5}',' _emoticon_ ',text)
	#convert unicode of newline to newline
	text = re.sub(r'\\n','',text)
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

def get_unique_value (list):
	unique_value = []
	for w in list:
		if w not in unique_value:
			unique_value.append(w)

	return unique_value


# def information_gain(label_data, frequency):
# 	total_sentence = label_data[0] + label_data[1] + label_data[2]
# 	frequency_class_a = float(label_data[0]) / float(total_sentence)
# 	frequency_class_b = float(label_data[1]) / float(total_sentence)
# 	frequency_class_c = float(label_data[2]) / float(total_sentence)
# 	prob_appear_class_a = float(frequency[0]) / float(total_sentence)
# 	prob_appear_class_b = float(frequency[1]) / float(total_sentence)
# 	prob_appear_class_c = float(frequency[2]) / float(total_sentence)
# 	prob_total_appear = prob_appear_class_a +prob_appear_class_b + prob_appear_class_c
# 	prob_unappear_class_a = float(label_data[0] - frequency[0]) / float(total_sentence)
# 	prob_unappear_class_b = float(label_data[1] - frequency[1]) / float(total_sentence)
# 	prob_unappear_class_c = float(label_data[2] - frequency[2]) / float(total_sentence)
# 	prob_total_unappear = prob_unappear_class_a + prob_unappear_class_b + prob_unappear_class_c


# 	part1 = (frequency_class_a*log10(frequency_class_a) + frequency_class_b*log10(frequency_class_b) + frequency_class_c*log10(frequency_class_c)) 
# 	# part2 = prob_total_appear * (prob_appear_class_a * log10(prob_appear_class_a) + prob_appear_class_b * log10(prob_appear_class_b) + prob_appear_class_c * log10(prob_appear_class_c))
# 	# part3 = prob_total_unappear * (prob_unappear_class_a * log10(prob_unappear_class_a) + prob_unappear_class_b * log10(prob_unappear_class_b) + prob_unappear_class_c * log10(prob_unappear_class_c))

# 	return log10(frequency[0])


list_of_data = []
# list_of_feature = []
list_of_label = []
list_of_comment = []
processed_data = []
bag_of_feature  = []
list_of_unique_word = []
# list_of_all_word = []
appear_label_jawab = []
appear_label_baca = []
appear_label_abaikan = []
count_jawab = 0
count_baca = 0
count_abaikan = 0

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
			list_of_label.append(label)
			list_of_comment.append(processed_comment)

			wordstring = processed_comment
			wordlist = wordstring.split()
			for w in get_unique_value(wordlist):
				# list_of_all_word.append(w)
				if w not in list_of_unique_word:
					list_of_unique_word.append(w)
					appear_label_jawab.append(0)
					appear_label_baca.append(0)
					appear_label_abaikan.append(0)
				if label == "jawab":
					appear_label_jawab[list_of_unique_word.index(w)] += 1
					count_jawab += 1
				elif label == "baca":
					appear_label_baca[list_of_unique_word.index(w)] += 1
					count_baca += 1
				elif label == "abaikan":
					appear_label_abaikan[list_of_unique_word.index(w)] += 1
					count_abaikan += 1

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

# wordfreq = []
# for w in list_of_unique_word:
#     wordfreq.append(list_of_all_word.count(w))

# print("List \n" + str(list_of_unique_word))
# print("Frequencies \n" + str(wordfreq))
# print(str(appear_label_jawab))
# print(str(appear_label_baca))
# print(str(appear_label_abaikan))

# print "Total Kalimat = ", len(list_of_data)
# print "Total Kata = ", len(list_of_unique_word)
# print "Total Label Jawab = ", jawab
# print "Total Label Baca = ", baca
# print "Total Label Abaikan = ", abaikan

fitur_frequency = []
for i in range(len(list_of_unique_word)):
	text_atribute = (appear_label_jawab[i], appear_label_baca[i], appear_label_abaikan[i], list_of_unique_word[i])
	fitur_frequency.append(text_atribute)

label_distribution = [jawab, baca, abaikan]

# print label_distribution
# print fitur_frequency

p = Popen(['java', '-jar', 'INLPPreproses.jar', 'formalization', "Aku yg seorang kapiten punya pedang bs jugaaaaa"], stdout=PIPE, stderr=STDOUT)
result = ""
for line in p.stdout:
	print line
	arr_token = line.decode("ascii", "replace").split(" ")    
	for token in arr_token:
		if token not in result and token != '_url_' and len(token) > 0:
			result += token + " "

text = result

# text = unicode(text, 'utf-8')
# text = text.encode('unicode_escape')
# #convert emoticon
# text = re.sub(r'\\U000[^\s]{5}',' _emoticon_ ',text)

print text

# print("Pairs\n" + str(zip(wordlist, wordfreq)))

# print("String\n" + wordstring +"\n")
# print("List\n" + str(wordlist) + "\n")
# print("Frequencies\n" + str(wordfreq) + "\n")
# print("Pairs\n" + str(zip(wordlist, wordfreq)))