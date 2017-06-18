import csv
import re

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

		# processed_comment = pre_process(comment)
		# # processed_comment = nltk.word_tokenize(comment)
		# print feature_extraction(nltk.word_tokenize(processed_comment))
		# bag_of_feature += feature_extraction(nltk.word_tokenize(processed_comment))
		# # list_of_feature.append(feature_extraction(processed_comment))
		list_of_label.append(label)
		list_of_comment.append(comment)

for index in range(90):
	#convert to unicode
	teststring = unicode(list_of_comment[index], 'utf-8')

	#encode it with string escape
	teststring = teststring.encode('unicode_escape')

	teststring = re.sub(r'\\U000[^\s]{5}',' _emoticon_ ',teststring)
	teststring = re.sub(r'\\n','',teststring)

	print list_of_comment[index]
	print teststring
	print re.sub(r"\\","","aabcd \aaa")