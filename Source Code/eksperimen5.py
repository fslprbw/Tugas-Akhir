import csv
import nltk
# from BaselineBok import BaselineBok
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Dense, Input, Conv1D, Embedding, MaxPooling1D, Flatten, Dropout, GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from keras import regularizers
from keras.constraints import max_norm
from keras.utils import np_utils
import numpy as np
import gensim
# import glove
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import random
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from collections import Counter
import csv
from sklearn.externals import joblib
# from imblearn.over_sampling import SMOTE, RandomOverSampler

EMBEDDING_DIM = 500 # / 400
MAX_SEQUENCE_LENGTH = 50 # ?? / 200
FILTER_SIZE = (3,4,5)
NUM_FILTERS = 100 #100 - 600 / 100
DROPOUT_PROB = (0.5, 0.5)
MAXNORM = 3
EMBEDDING = 'old' # 'old' | 'new'

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 10

def printToCSV (data_list, filename):
    with open('../Resource/'+filename+'.csv', 'w') as csvfile:
        fieldnames = ['no', 'word']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()

        for index in range(len(data_list)):
            writer.writerow({'no':index, 'word':data_list[index]})

def shuffle_weights(model, weights=None):

    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

if __name__ == '__main__':

    MODEL_TYPE = 'non-static'
    file_csv = '../Resource/all_preprocessed.csv'

    raw_sentences = []
    labels = []

    with open(file_csv, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            if line["label"] != "x":
                raw_sentences.append(line["content"])
                labels.append(line["label"])

    labels_index = {'jawab': 0, 'baca': 1, 'abaikan': 2}
    reverse_labels_index = {0: 'jawab', 1: 'baca', 2: 'abaikan'}
    labels = [labels_index[label] for label in labels]    

    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(raw_sentences)
    sequences = tokenizer.texts_to_sequences(raw_sentences)
    word_index = tokenizer.word_index

    # joblib.dump(tokenizer, 'wecnn_tokenizer.pkl')

    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    X = data
    Y = np_utils.to_categorical(np.asarray(labels))
    labels = np.array(labels)

    w2v_model = gensim.models.Word2Vec.load('best_model_we')

    embeddings_index = dict(zip(w2v_model.wv.index2word, w2v_model.wv.syn0))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))

    if MODEL_TYPE == 'static':
        trainable = False 
    else: 
        trainable = True

    if MODEL_TYPE == 'rand':        
        embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=trainable)
    else:
        embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    dropout_layer_1 = Dropout(DROPOUT_PROB[0])
    conv_list = []
    for index, filtersize in enumerate(FILTER_SIZE):
        nb_filter = NUM_FILTERS
        conv = Conv1D(filters=NUM_FILTERS, kernel_size=filtersize, activation='relu')(sequence_input)
        conv = GlobalMaxPooling1D()(conv)
        conv_list.append(conv)

    out = Concatenate()(conv_list) if len(conv_list) > 1 else conv_list[0]
    dropout_layer_2 = Dropout(DROPOUT_PROB[1])
    activation_layer = Dense(EMBEDDING_DIM, activation="relu")
    model_output = Dense(len(labels_index), activation="softmax", kernel_constraint=max_norm(MAXNORM))

    graph = Model(sequence_input, out)

    early_stopping = EarlyStopping(monitor='val_acc', patience=1)

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_confusion_matrix = 0

    # ========================================

    num_folds =10
    size = len(labels)
    subset_size = size/num_folds
    sum_acc = 0
    result_label = []

    model_seq = Sequential()
    model_seq.add(embedding_layer)
    model_seq.add(dropout_layer_1)
    model_seq.add(graph)
    model_seq.add(dropout_layer_2)
    model_seq.add(model_output)

    model_seq.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model_seq.save_weights('model.h5')

    initial_weights = model_seq.get_weights()

    # text = "alhamdulillah _tandatitik_ terima kasih _emotpos_ _emotpos_ asukon"
    # sentence = []
    # sentence.append(text)
    
    # text_sequences = tokenizer.texts_to_sequences(sentence)

    # print text_sequences

    # we_vector = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    for index in range(10):

        model_seq.load_weights('model.h5')
  
        if index < size % num_folds:
            test_start = index*(subset_size+1)
            test_finish = test_start + subset_size + 1
        else:
            test_start = (index*subset_size) + (size % num_folds)
            test_finish = test_start + subset_size
        
        test_labels = labels[test_start:test_finish]
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

        print "Fold ke : ", index
        print "Train size = ", len(train_comment_round)
        print "Test size = ", len(test_comment_round)
        # for epoch in range(NUM_EPOCHS):
        #     print('Fold: {}'.format(i+1))
        #     print('Epoch: {}'.format(epoch+1))

        model_seq.fit(train_comment_round, train_label_round, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)          

        y_pred = model_seq.predict_classes(test_comment_round, batch_size=512)
        fold_accuracy = accuracy_score(test_labels,y_pred)

        result_label = np.concatenate((result_label, y_pred))
        sum_acc += fold_accuracy

    print confusion_matrix(labels ,result_label, labels=[0, 1, 2])
    print sum_acc/num_folds
    printToCSV(result_label, "hasil_CNN_test2_10")

        # model_seq.fit(X, Y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)  
        # model_seq.save('WE-CNN_model_10epoch.h5')

        # result = model_seq.predict_classes(we_vector, batch_size=512)        
        # reverse_labels_index = {0: 'jawab', 1: 'baca', 2: 'abaikan'}
        # print [reverse_labels_index[label] for label in result]
