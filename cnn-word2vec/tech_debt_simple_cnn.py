#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np




class tech_debt_simple_cnn:
    
    def get_data(self):
        d1=pd.read_csv("D:/RIT/GA-TECHNICAL DEBTS/rudimentary-stages/data/all/sql12.csv")
        d2=pd.read_csv("D:/RIT/GA-TECHNICAL DEBTS/rudimentary-stages/data/all/apache-ant-1.7.0.csv")
        d3=pd.read_csv("D:/RIT/GA-TECHNICAL DEBTS/rudimentary-stages/data/all/apache-jmeter-2.10.csv")
        d4=pd.read_csv("D:/RIT/GA-TECHNICAL DEBTS/rudimentary-stages/data/all/argouml.csv")
        d5=pd.read_csv("D:/RIT/GA-TECHNICAL DEBTS/rudimentary-stages/data/all/columba-1.4-src.csv")
        d6=pd.read_csv("D:/RIT/GA-TECHNICAL DEBTS/rudimentary-stages/data/all/emf-2.4.1.csv")
        d7=pd.read_csv("D:/RIT/GA-TECHNICAL DEBTS/rudimentary-stages/data/all/hibernate-distribution-3.3.2.GA.csv")
        d8=pd.read_csv("D:/RIT/GA-TECHNICAL DEBTS/rudimentary-stages/data/all/jEdit-4.2.csv")
        d9=pd.read_csv("D:/RIT/GA-TECHNICAL DEBTS/rudimentary-stages/data/all/jfreechart-1.0.19.csv")
        frames=[d1,d2,d3,d4,d5,d6,d7,d8,d9]
        data=pd.concat(frames)
        #print(data.head())
        
        
        '''Remove punctuations'''       
        import re
        import string
        def remove_punct(text):
            text_nopunct = ''
            text_nopunct = re.sub('['+string.punctuation+']', '', text)
            return text_nopunct
        data['clean_comment'] = data['Abstract'].apply(lambda x: remove_punct(x))
        
        '''Finding max length of comment in our dataset -11220 after punct - 9325'''
        mx_dct = {c: data[c].map(lambda x: len(str(x))).max() for c in data.columns}
        print(pd.Series(mx_dct).sort_values(ascending =False)) #print maxlen
                
        #print(data.head()) #see clean comments column
        
        comments=data[['clean_comment']]
        labels=data[['label']]
        labels_list=[]
        for i, row in labels.iterrows():
            labels_list.append(row['label'])

        comments_list=[]
        for i, row in comments.iterrows():
            comments_list.append(row['clean_comment'])
            
        #print("\n\n", comments_list[:10], "\n\n\n")
        #print("\n\n",labels_list[:10])
        
        return comments_list,labels_list,data
    
    def preprocess(self):
        
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.utils import to_categorical
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        c, l,data= self.get_data()
        #print("\n\n", c[:20], "\n\n\n")
        #print("\n\n",l[:20])
        
        NUM_WORDS=2000 #must be 9325 actually, also do we include \r in filters? 
        tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                              lower=True)
        tokenizer.fit_on_texts(c)
        X = tokenizer.texts_to_sequences(c)
        X,l= np.array(X), np.array(l)
        word_index = tokenizer.word_index
        #print('\nFound %s unique tokens.' % len(word_index))       

        '''print word  index example'''
        print("\n\n", list(word_index.items())[:25])
      
        '''verify length of   word index = unique words found '''
        #print(len(word_index))
        
        label_encoder = LabelEncoder()
        vec = label_encoder.fit_transform(l)
        y=to_categorical(vec)
        
        X = pad_sequences(X,maxlen=NUM_WORDS)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
        data = {}
        data["X_train"] = X_train
        data["X_val"]= X_val
        data["y_train"] = y_train
        data["y_val"] = y_val
        data["tokenizer"] = tokenizer
        data["int2label"] =  {0: "no", 1: "yes"}
        data["label2int"] = {"no": 0, "yes": 1}
        
        #print(data)
        print('\n\nShape of X train and X validation tensor:', X_train.shape,X_val.shape)
        print('\n\nShape of label train and validation tensor:', y_train.shape,y_val.shape)
        return word_index,X_train, X_val, y_train, y_val
    
    def get_embeddings(self):
        
        import gensim
        from gensim.models import Word2Vec
        from gensim.utils import simple_preprocess

        from gensim.models.keyedvectors import KeyedVectors
        word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        
        #load pre trained weights
        print(word_vectors)
        
        word_index,X_train, X_val, y_train, y_val=self.preprocess()
        print('\nFound %s unique tokens in embeddings func.' % len(word_index))       
        
        EMBEDDING_DIM=300
        NUM_WORDS=2000
        vocabulary_size=min(len(word_index)+1,NUM_WORDS)
        embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i>=NUM_WORDS:
                continue
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
        print(embedding_matrix)
        
        sequence_length = X_train.shape[1]
        print(sequence_length)
        
        return embedding_matrix, word_index, X_train, X_val, y_train, y_val

    def build_model(self):
        
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
        from tensorflow.keras.models import Model, Sequential
        from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate,GlobalMaxPooling1D
        from tensorflow.keras.layers import Reshape, Flatten
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import Model
        from tensorflow.keras import regularizers
        from tensorflow.keras import optimizers
        
        embedding_matrix,word_index, X_train, X_val, y_train, y_val =self.get_embeddings()
        print("\n\nShape of embedding matrix: ",embedding_matrix.shape)
        print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        
        EMBEDDING_DIM=300
        NUM_WORDS=2000
        vocabulary_size=min(len(word_index)+1,NUM_WORDS)
        embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=False)
        
        sequence_length =2000
        #have to put these in init
        batch_size = 32 
        num_epochs = 10

        #model parameters
        num_filters = 256 #put this later 
        weight_decay = 1e-4
        sequence_input = Input(shape=(sequence_length,))
        embedded_sequences = embedding_layer(sequence_input)
        print(embedded_sequences, "\n\n\n")
        
        convs = []
        filter_sizes = [2,3,4,5,6]
        for filter_size in filter_sizes:
            l_conv = Conv1D(filters=64, 
                             kernel_size=filter_size, 
                            activation='relu')(embedded_sequences)
            l_pool = GlobalMaxPooling1D()(l_conv)
            convs.append(l_pool)
        l_merge = concatenate(convs, axis=1)

        x = Dropout(0.1)(l_merge)  
        x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        preds = Dense(2, activation='softmax')(x)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)      
        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
        print(model.summary())
        
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
        callbacks_list = [early_stopping]
        
        #train model
        def train(self):
            hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, 
                validation_split=0.2, shuffle=True, verbose=2)
       
            model_name="v1-cnn-word2vec"      

            scores = model.evaluate(X_val, y_val, verbose=0)
            print("Model Accuracy: %.2f%%" % (scores[1]*100))

            #save model 
            if not os.path.isdir("results"):
                os.mkdir("results")
            if not os.path.isdir("logs"):
                os.mkdir("logs")
            if not os.path.isdir("data"):
                os.mkdir("data")
            model_name="cnn-imdb"
            model.save(os.path.join("results", model_name) + ".h5")
            return

        def show(self):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(ist.history['accuracy'], lw=2.0, color='b', label='train')
            plt.plot(ist.history['val_accuracy'], lw=2.0, color='r', label='val')
            plt.title('CNN label')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend(loc='upper left')
            plt.show()
            
            
            plt.figure()
            plt.plot(ist.history['loss'], lw=2.0, color='b', label='train')
            plt.plot(ist.history['val_loss'], lw=2.0, color='r', label='val')
            plt.title('CNN label')
            plt.xlabel('Epochs')
            plt.ylabel('Cross-Entropy Loss')
            plt.legend(loc='upper right')
            plt.show()
        
        #self.train()
        #self.show()
        
        def test(self, model):
            import tensorflow as tf
            dtest=pd.read_csv("D:/RIT/GA-TECHNICAL DEBTS/rudimentary-stages/data/all/jruby-1.4.0.csv")
            c=dtest[['Abstract']]
            l=dtest[['label']]
            labels_test=[]
            for i, row in l.iterrows():
                labels_test.append(row['label'])

            comments_test=[]
            for i, row in c.iterrows():
                comments_test.append(row['Abstract'])
                
                
            NUM_WORDS=2000
            tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                                  lower=True)
            tokenizer.fit_on_texts(comments_test)
            Xtest = tokenizer.texts_to_sequences(comments_test)
            Xtest,ll= np.array(Xtest), np.array(labels_test)
            Xtest = pad_sequences(Xtest,maxlen=2000)
            
            label_encoder = LabelEncoder()
            vec1 = label_encoder.fit_transform(ll)
            ytest=to_categorical(vec1)
            
            scores1 = model.evaluate(Xtest, ytest, verbose=0)
            print("Test Accuracy: %.2f%%" % (scores1[1]*100))
            
            y_pred=model.predict(Xtest)
            
            cm=tf.math.confusion_matrix(labels=tf.argmax(ytest, 1), predictions=tf.argmax(y_pred, 1))
            
            print(cm)
            return
        
        #self.show()
        
    
        
ob=tech_debt_simple_cnn()
ob.build_model()
        


 




