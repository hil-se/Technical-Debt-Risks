import tensorflow as tf
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#pd.set_option('display.max_colwidth', None)
import glob
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load dataframes from data dir
filenames = list(glob.glob("data/*"))
dataframes = [pd.read_csv(file) for file in filenames]

metadata = pd.concat(dataframes)
print("\nReading set\n*************************************************************\n" )
print(metadata.head(10))
print("\nForming labels\n...............\n")
metadata['label'] = np.where(metadata.classification == 'WITHOUT_CLASSIFICATION', 0, 1)
print(metadata.head())

print("\nDistribution of labels:\n\n")
print(metadata['label'].value_counts())
print(metadata.projectname.unique())

print("\n*********************************** Current Train set *******************************************\n")
train_lst = ['apache-ant-1.7.0' ,'apache-jmeter-2.10', 'argouml' ,'columba-1.4-src'
 'emf-2.4.1', 'hibernate-distribution-3.3.2.GA', 'jEdit-4.2',
 'jfreechart-1.0.19' , 'sql12']

df =  metadata[metadata['projectname'].isin(train_lst)]

print(df.isnull().sum(),"\n")
print(df.info(), "\n")


'''
1] Preprocessing:
1. Remove tags, punctuations, stop words, special characters and return X_clean and y as np arrays
2. Split data in train and test. Remember Xclean and y should reflect metadata used for training
def remove_tags(text):

'''
def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')

    return TAG_RE.sub('', text)
    
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)
    #print(sentence)
    return sentence

def prepare_data(_df):
    X_clean = []
    sentences = list(_df['commenttext'])

    for sen in sentences:
        X_clean.append(preprocess_text(sen))

    print(X_clean[:3], "\n\n*************************")
    X_clean = np.array(X_clean)
    y = np.array(_df.label)

    print("Cleaned corpus shape:", X_clean.shape, y.shape)
    return X_clean, y

def split_data(df):
    X_clean, y = prepare_data(df)
    print(X_clean[:3], "\n****kkk****\n")
    
    X_train, X_val, y_train, y_val = train_test_split(X_clean, y, test_size=0.20, random_state=42)
    print("Train set:",X_train.shape, y_train.shape)
    print("Validation set:", X_val.shape, X_val.shape)
    return X_train, X_val, y_train, y_val

    #call preprocess
    print("Preprocessing and splitting data\n******************************************\n")
    X_train, X_val, y_train, y_val = split_data(df)




X_train, X_val, y_train, y_val = split_data(df)
print(X_train.shape)

'''
2] Tokenize the splitted data and convert them intp sequences and add post padding.
Final output stored as X_train_padded, X-val_padded and y_train and y_val

'''
print("\n**********************************************PRE-PROCESSING DATA*********************************************\n")
def create_features(X_train, X_val, y_train, y_val):
    
    #Step 2: Use keras to tokenize words and find word.index'length for getting number of unique words i.e vocab size

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)

    tokenizer.fit_on_texts(X_train)

    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print('VOCAB SIZE: Found %s unique tokens.' % len(word_index))

    # Step 3: Embed the sentences into numbers using text to sequences

    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_valid = tokenizer.texts_to_sequences(X_val)

    # Step 4: next step is to find the number of words in the longest sentence and then to apply 
    #padding to the sentences having shorter lengths than the length of the longest sentence

    from nltk.tokenize import word_tokenize
    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence = max(X_train, key=word_count)
    length_long_sentence = len(word_tokenize(longest_sentence))
    print("Longest sentence length: ",  length_long_sentence)
    
    print("*******************************************\n")
    
    #Step 5: Pad zeros at the end using the length of the longest word 
    X_train_padded = pad_sequences(sequences_train,length_long_sentence, padding='post')
    X_val_padded = pad_sequences(sequences_valid,padding='post', maxlen = length_long_sentence)


    print('\nShape of X train and X validation tensor:', X_train_padded.shape,X_val_padded.shape)
    print('\nShape of label train and validation tensor:', y_train.shape,y_val.shape)
    
    return X_train_padded, X_val_padded, y_train, y_val, vocab_size, word_index, length_long_sentence

#call tokenize
X_train_padded, X_val_padded, y_train, y_val, vocab_size, word_index, length_long_sentence= create_features(X_train, X_val, y_train, y_val)

print("Padded data:\n", X_train_padded)

#********************************************************************PATH TO TRAINED EMBEDDINGS*****************************************
print("********************************************************************BUILDING EMBEDDINGS*****************************************")
'''3] Build embedidng matrix : Word2vec and glove (needs fixes)'''
def build_embedding_matrix_word2vec(vocab_size, word_index):
    import gensim
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess

    from gensim.models.keyedvectors import KeyedVectors
    path_to_word2vec = 'fast-text/crawl-300d-2M-subword.vec'
    word_vectors = KeyedVectors.load_word2vec_format(path_to_word2vec)
    
    
    EMBEDDING_DIM= 300
    vocab_size = vocab_size
    
    #Size of embed matrix must be = vocab sizeV * dimension of embeding
    #Step 6: Generate an embedding matrix to get embeddings representaion of words in our corpus
    #Our embedding_matrix now contains pretrained word embeddings for the words in our corpus.
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i>=vocab_size:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
    
    return embedding_matrix, word_vectors

embedding_matrix, word_vectors =  build_embedding_matrix_word2vec(vocab_size, word_index)

print("\n\nEmbedding matrix shape:\n*******************************************\n",embedding_matrix.shape)


print("\n**********************************************CHECKING CUSTOM WEIGHTS NOW\n*********************************************\n")
'''4] Weighted loss func to handle class imbalance'''

def sklearn_weighted_loss(y_train):
    # Calculate the weights for each class so that we can balance the data
    #The minority class will have a higher class weight
    from sklearn.utils import class_weight
    weights = class_weight.compute_class_weight('balanced',
                                                np.unique(y_train),
                                                y_train)
    print("Resultant weight:", weights)
    sklearn_weight = {0: weights[0], 1: weights[1]}
    return sklearn_weight

def weighted_loss(y_train):
    '''Link: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data'''

    neg, pos = np.bincount(y_train)
    total = neg + pos
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight

custom_class_weights = weighted_loss(y_train)
print("Custom weights:", custom_class_weights)
#or use 
sklearn_weights= sklearn_weighted_loss(y_train)




from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from tensorflow.keras.layers import Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend 
EMBEDDING_DIM = 300

print("\n********************************************DEFINE CNN 2D MODEL***********************************************\n")


filter_sizes = [1,2,3,4,5,6]
num_filters = 128
drop = 0.1
EMBEDDING_DIM = 300 

deep_inputs = Input(shape=(length_long_sentence,))
embedding = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=length_long_sentence, trainable=False)(deep_inputs)
reshape = Reshape((length_long_sentence,EMBEDDING_DIM,1))(embedding)

conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu', kernel_regularizer = regularizers.l2(0.01))(reshape)
conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu', kernel_regularizer = regularizers.l2(0.01))(reshape)
conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu', kernel_regularizer = regularizers.l2(0.01))(reshape)
conv_3 = Conv2D(num_filters, (filter_sizes[3], EMBEDDING_DIM),activation='relu', kernel_regularizer = regularizers.l2(0.01))(reshape)
conv_4 = Conv2D(num_filters, (filter_sizes[4], EMBEDDING_DIM),activation='relu', kernel_regularizer = regularizers.l2(0.01))(reshape)
conv_5 = Conv2D(num_filters, (filter_sizes[5], EMBEDDING_DIM),activation='relu', kernel_regularizer = regularizers.l2(0.01))(reshape)

maxpool_0 = MaxPooling2D((length_long_sentence - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
maxpool_1 = MaxPooling2D((length_long_sentence - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
maxpool_2 = MaxPooling2D((length_long_sentence - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)
maxpool_3 = MaxPooling2D((length_long_sentence - filter_sizes[3] + 1, 1), strides=(1,1))(conv_3)
maxpool_4 = MaxPooling2D((length_long_sentence - filter_sizes[4] + 1, 1), strides=(1,1))(conv_4)
maxpool_5 = MaxPooling2D((length_long_sentence - filter_sizes[5] + 1, 1), strides=(1,1))(conv_5)


merged_tensor = concatenate([maxpool_0,maxpool_1,maxpool_2, maxpool_3,maxpool_4,maxpool_5], axis=3)

#batch_normalization = tf.keras.layers.BatchNormalization()(merged_tensor)
flatten = Flatten()(merged_tensor)
reshape = Reshape((4*num_filters,))(flatten)
#dropout = Dropout(drop)(flatten)

#add dense layers here, RELU same i/p,  o/p size
cov0_dense =  Dense(units=64, activation='relu', kernel_regularizer = regularizers.l2(0.01))(flatten)
cov1_dense =  Dense(units=64, activation='relu', kernel_regularizer = regularizers.l2(0.01))(cov0_dense)
cov2_dense =  Dense(units=64, activation='relu', kernel_regularizer = regularizers.l2(0.01))(cov1_dense)
dense_1 =  Dense(units=64, activation='relu', kernel_regularizer = regularizers.l2(0.01))(cov2_dense)

output = Dense(units=1, activation='sigmoid', kernel_regularizer = regularizers.l2(0.01))(dense_1)
#p(c=1)+ p(c=0) = 1
# this creates a model that includes
cnn2d_model = Model(deep_inputs, output)


sgd = SGD(learning_rate=0.0095)
cnn2d_model.compile(optimizer = sgd , loss='binary_crossentropy', metrics=['accuracy']) 
#use tf.keras.losses.BinaryCrossentropy(from_logits=True) in loss
callbacks = [EarlyStopping(monitor='loss',  min_delta=0.0000000001)]
print(cnn2d_model.summary())


cnn2d_history = cnn2d_model.fit(X_train_padded, y_train, batch_size=32, epochs=100, validation_split=0.15, 
                                class_weight = sklearn_weights,shuffle = True)


print("\n**********************************************OUTPUT ACCURACY****************************************************************\n")

score = cnn2d_model.evaluate(X_val_padded, y_val, verbose=1)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])



print("\n\n\n\n")
print("\n**********************************************METRICS EVALUATION - VALIDATION SET*********************************************\n")
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

y_pred1 = cnn2d_model.predict(X_val_padded)
ypred_ = [1 if pred[0]>0.5 else 0 for pred in y_pred1]


print("Precision:", round(precision_score(y_val, ypred_ , average="binary", pos_label = 1),2))
print("Recall:", round(recall_score(y_val, ypred_, average="binary", pos_label = 1),2))
print("F1 Score:", round(f1_score(y_val, ypred_ , average="binary", pos_label = 1),2)) 


print("\n**********************************************METRICS EVALUATION - TEST SET*********************************************\n")
def create_features_test(X_train, X_test):
    
    #Step 2: Use keras to tokenize words and find word.index'length for getting number of unique words i.e vocab size

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)

    tokenizer.fit_on_texts(X_train)

    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print('VOCAB SIZE: Found %s unique tokens.' % len(word_index))

    # Step 3: Embed the sentences into numbers using text to sequences

    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)

    # Step 4: next step is to find the number of words in the longest sentence and then to apply 
    #padding to the sentences having shorter lengths than the length of the longest sentence

    from nltk.tokenize import word_tokenize
    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence = max(X_train, key=word_count)
    length_long_sentence = len(word_tokenize(longest_sentence))
    print("Longest sentence length: ",  length_long_sentence)
    
    print("**********************")
    
    #Step 5: Pad zeros at the end using the length of the longest word 
    X_test_padded = pad_sequences(sequences_test,padding='post', maxlen = length_long_sentence)


    print('Shape of X test tensor:', X_test_padded.shape)
    
    return X_test_padded

test_set = metadata[~ metadata['projectname'].isin(train_lst)]
X_test_clean, y_test = prepare_data(test_set)
X_test_padded = create_features_test(X_train, X_test_clean)

y_pred_test = cnn2d_model.predict(X_test_padded)

predictions = [1 if pred[0]>0.5 else 0 for pred in y_pred_test]
print("F1 Score:", round(f1_score(y_test, predictions , average="binary", pos_label = 1),2))
print("Precision:", round(precision_score(y_test, predictions , average="binary", pos_label = 1),2))
print("Recall:", round(recall_score(y_test, predictions , average="binary", pos_label = 1),2))

print(test_set.projectname.unique())


print("\n**********************************************  GRAPHS  *********************************************\n")
