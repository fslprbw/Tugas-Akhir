import os, sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy as np
import scipy as scipy

from subprocess import Popen, PIPE, STDOUT
import re
import nltk
import numpy as np

from sklearn.externals import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

app = QApplication(sys.argv)
win = QWidget()

fbox = QFormLayout()

l0 = QLabel("Akun Instagram")
acc = QLineEdit()

l1 = QLabel("Komentar Instagram")
nm = QLineEdit()

l3 = QLabel("Praproses")
ck_box = QVBoxLayout()
ck1 = QCheckBox("Konversi URL")
ck2 = QCheckBox("Konversi Hashtag")
ck3 = QCheckBox("Konversi Mention")
ck4 = QCheckBox("Konversi Tanda Baca")
ck5 = QCheckBox("Konversi Emoticon")
ck6 = QCheckBox("Konversi Angka")
ck7 = QCheckBox("Formalisasi Kata + Konversi Daerah")
ck8 = QCheckBox("Penghapusan Stopword")

l4 = QLabel("Algoritma Pembelajaran")
r_box = QVBoxLayout()
r0 = QRadioButton("Bseline SVM")
r1 = QRadioButton("Unigram + SVM")
r2 = QRadioButton("Unigram + Decision Tree")
r3 = QRadioButton("Unigram + Naive Bayes")
r4 = QRadioButton("Word Embedding + SVM")
r5 = QRadioButton("Word Embedding + CNN")

b1 = QPushButton("Submit")

l_praprosesed = QLabel("")
l_result = QLabel("")

def convert_emoticon (text):

   emot_positif = ["u0001f600","u0001f601","u0001f602","u0001f923","u0001f603","u0001f604","u0001f605","u0001f606","u0001f607","u0001f609","u0001f60a","u0001f60b","u0001f60e","u0001f60d","u0001f60e","u0001f618","u0001f617","u0001f618","u0001f619","u0001f61a","u0001f63a","u0001f642","u0001f917","u0001f929","u0001f44a","u0001f44c","u0001f44d","u0001f44f","u0001f495","u0001f496","u0001f49c","u0001f49e",":)",":-)",":D",":-D",":*"]
   emot_negatif = ["u0001f608","u0001f641","u0001f616","u0001f61e","u0001f61f","u0001f624","u0001f622","u0001f62d","u0001f626","u0001f627","u0001f628","u0001f629","u0001f92f","u0001f62c","u0001f630","u0001f631","u0001f633","u0001f92a","u0001f635","u0001f63f","u0001f621","u0001f620","u0001f92c","u0001f494",":(",":-(",";(",";-("]
   emot_netral = ["u0001f914","u0001f928","u0001f610","u0001f611","u0001f612","u0001f614","u0001f636","u0001f644","u0001f64b","u0001f64c","u0001f64f","u0001f68c","u0001f60f","u0001f623","u0001f625","u0001f62e","u0001f910","u0001f62f","u0001f62a","u0001f62b","u0001f634","u0001f60c","u0001f61b","u0001f61c","u0001f61d","u0001f924","u0001f612","u0001f613","u0001f614","u0001f615","u0001f643","u0001f911","u0001f632","u0001f1f0","u0001f1f5","u0001f334","u0001f338","u0001f34e","u0001f3b6","u0001f3ba","u0001f3fb","u0001f3fc","u0001f479","u0001f47b","u0001f483","u0001f48b","u0001f4e2","u0001f4e3"]
   result = ""

   if text.group(0).lower()[1:] in emot_positif:
      result = " _emot_pos_ "
   elif text.group(0).lower()[1:] in emot_negatif:
      result = " _emot_neg_ "
   else:
      result = " _emot_netral_ "

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

def pra_proses(text, poster):

   #turn emoticon to unicode
   text = unicode(text, 'utf-8')
   text = text.encode('unicode_escape')

   #convert unicode of newline to newline
   text = re.sub(r'\\n',' ',text)

   if ck1.isChecked():
      # Convert www.* or https?://* to URL
      text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' _url_ ',text)
   if ck2.isChecked():
      #Replace #word with word
      text = re.sub(r'#([^\s]+)',' _hashtag_ ', text)
   if ck3.isChecked():
      # Convert @username to AT_USER
      text = re.sub('@'+poster,' _mentionpemilik_ ',text)
      text = re.sub('@[^\s]+',' _mentionteman_ ',text) 
   if ck4.isChecked():
      #Convert mark
      text = re.sub('[/]+', ' ', text)
      text = re.sub('[&]+', ' ', text)
      text = re.sub('[,]+', ' ', text)
      text = re.sub('[.]+', ' _tandatitik_ ', text)
      text = re.sub('[?]+', ' _tandatanya_ ', text)
      text = re.sub('[!]+', ' _tandaseru_ ', text)
   if ck5.isChecked():
      #convert emoticon and symbol
      text = re.sub(r'\\U000[^\s]{5}',convert_emoticon,text)
      # text = re.sub(r'\\u[\d][^\s]{3}',' _emoticon_ ',text)
   if ck6.isChecked():
      #convert digit
      text = re.sub('[\d]+', ' _angka_ ', text)
   if ck7.isChecked():
      text = formalization(text+"`")
   if ck8.isChecked():
      text = remove_stopword(text+"`")

   text = re.sub('[\s]+', ' ', text)
   #Convert to lower case
   text = ''.join(text).lower()

   return text

def classify(text):

   out = ""
   word_list = joblib.load('unigram_wordlist.pkl')
   
   comment_vector = np.zeros((1, len(word_list)))

   word = nltk.word_tokenize(text)

   for index in range(len(word_list)):
      if word_list[index] in word:
         comment_vector[0][index] += 1

   if r0.isChecked():
      clf = joblib.load('baseline-SVM_model.pkl')
      out = clf.predict(comment_vector)[0]
   elif r1.isChecked():
      clf = joblib.load('unigram-SVM_model.pkl')
      out = clf.predict(comment_vector)[0]
   elif r2.isChecked():
      clf = joblib.load('unigram-DT_model.pkl')
      out = clf.predict(comment_vector)[0]
   elif r3.isChecked():
      clf = joblib.load('unigram-NB_model.pkl')
      out = clf.predict(comment_vector)[0]
   elif r4.isChecked():
      word_vector = joblib.load('we_word_vector.pkl')
      we_vector = []

      word_in_sentence = 0
      vector_size = len(word_vector[1][0])
      sum_vector = [0] * vector_size
      average_vector = [0] * vector_size
      for word_index in range(len(comment_vector)):
         word_count = comment_vector[0][word_index]
         if (word_count >= 1):
            word_in_sentence += word_count
            if word_list[word_index] in word_vector[0]:
               for vector_index in range(vector_size):
                  sum_vector[vector_index] += word_count*(word_vector[1][word_index][vector_index])

      for vector_index in range(vector_size):
         if word_in_sentence != 0:
            average_vector[vector_index] = sum_vector[vector_index]/word_in_sentence

      we_vector.append(average_vector)

      clf = joblib.load('WE-SVM_model.pkl')
      out = clf.predict(we_vector)[0]
   elif r5.isChecked():
      sentence = []
      sentence.append(text)
      
      tokenizer =  joblib.load('wecnn_tokenizer.pkl')
      text_sequences = tokenizer.texts_to_sequences(sentence)

      we_vector = pad_sequences(text_sequences, maxlen=50, padding="post", truncating="post")

      model_seq = load_model('WE-CNN_model.h5')

      result = model_seq.predict_classes(we_vector, batch_size=512)        
      reverse_labels_index = {0: 'jawab', 1: 'baca', 2: 'abaikan'}
      out = [reverse_labels_index[label] for label in result][0]

   return out

def b1_clicked():
   comment = str(nm.text())
   account = str(acc.text())

   pre_processed_comment = pra_proses(comment, account)
   result = classify(pre_processed_comment)

   l_praprosesed.setText(pre_processed_comment)
   l_result.setText(result)

def window():
 
   fbox.addRow(l0,acc)   

   fbox.addRow(l1,nm)   

   ck_box.addWidget(ck1)
   ck_box.addWidget(ck2)
   ck_box.addWidget(ck3)
   ck_box.addWidget(ck4)
   ck_box.addWidget(ck5)
   ck_box.addWidget(ck6)
   ck_box.addWidget(ck7)
   ck_box.addWidget(ck8)
   ck_box.addStretch()
   fbox.addRow(l3,ck_box)

   r_box.addWidget(r0)
   r_box.addWidget(r1)
   r_box.addWidget(r2)
   r_box.addWidget(r3)
   r_box.addWidget(r4)
   r_box.addWidget(r5)
   r_box.addStretch()
   fbox.addRow(l4,r_box)

   fbox.addRow(b1)

   fbox.addRow(l_praprosesed)
   fbox.addRow(l_result)

   QObject.connect(b1,SIGNAL("clicked()"),b1_clicked)

   win.setLayout(fbox)
   
   win.setWindowTitle("PyQt")
   win.show()
   sys.exit(app.exec_())

if __name__ == '__main__':
   window()