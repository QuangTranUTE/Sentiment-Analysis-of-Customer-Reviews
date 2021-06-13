''' 
SENTIMENT ANALYSIS OF CUSTOMER REVIEWS
https://github.com/QuangTranUTE/Sentiment-Analysis-of-Customer-Reviews 
quangtn@hcmute.edu.vn

INSTRUCTIONS:
    + Run entire code: if you want to train your model from scratch. You can easily customize the model and stuff by changing hyperparameters put at the beginning of code parts (marked with comments NOTE: HYPERPARAM)
    + Run only Part 1 & Part 4: if you already trained (and saved) a model and want to do prediction (review analysis).
For other instructions, such as how to prepare your data, please see the github repository given above.

The code below have been successfully run on a system with:
Package         version
------------------------        
python          3.7.9
tensorflow      2.4.0
mosestokenizer  1.1.0
joblib          1.0.1
numpy           1.19.5
pandas          1.2.3
'''


# In[1]: PART 1. IMPORT AND FUNCTIONS
#region
import sys
from tensorflow import keras
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import tensorflow as tf
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
import numpy as np
import joblib
from mosestokenizer import MosesTokenizer, MosesDetokenizer
import gdown
import zipfile
import pandas as pd
import string
import re 
import unicodedata  
    
# Setup for GPU usage:
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# Declarations and functions for data preprocessing (used in Part 2 & Part 4)
eos_id = 0 # end-of-seq token id 
sos_id = 1 # start-of-seq token id
oov_id = 2 # out-of-vocab word id
def word_to_id(word, vocab_list):
    if word in vocab_list:
        return vocab_list.index(word) 
    else:
        return oov_id 
def id_to_word(id, vocab_list):
    return vocab_list[id]

#endregion


# In[2]: PART 2. LOAD AND PREPROCESS DATA
# Hyperparameters:
N_WORDS_KEPT = 150 # NOTE: HYPERPARAM. Number of words to keep in each sample (a review)             
min_occurrences = 4 # NOTE: HYPERPARAM. Each word appear many times in the dataset. We only keep the words that occur >= min_occurrences in the dataset. Amitabha 
# [NOTE: HYPERPARAM] Under-sampling and over-sampling at CODE LINE 88

#region
# LOAD DATA:
# Brief description of the data:
#   Crawled (from Vietname e-commerce websites) and labeled by Trần Gia Bảo, Trần Thị Tâm Nguyên, Hoàng Thị Cẩm Tú, Uông Thị Thanh Thủy.           
#   About 100k reviews from main categories: Clothing, Shoes, Bags, Luggage, Watches, and other Fashion accessories
#   More info: see `datasets` on https://github.com/QuangTranUTE/Sentiment-Analysis-of-Customer-Reviews 

data_file_path = r'datasets/customer_reviews.csv'
new_download = True
if new_download:
    url_data = 'https://drive.google.com/u/0/uc?id=' ########## TO BE ADDED
    download_output = 'temp.zip'
    gdown.download(url_data, download_output, quiet=False)
    with zipfile.ZipFile(download_output, 'r') as zip_f:
        zip_f.extractall(data_file_path)
raw_data = pd.read_csv(data_file_path)
print('\nData info: ',)
print(raw_data.info())
print('\nSome reviews: \n', raw_data.head(3))
print('\nData length:',len(raw_data))

# PREPROCESS DATA:
# [NOTE: HYPERPARAM] Under-sampling and over-sampling classes to reduce imbalance:
data_class0=raw_data[(raw_data.new_label==0)]
data_class1=raw_data[(raw_data.new_label==1)][:30000] # Under-sampling
data_class2=raw_data[(raw_data.new_label==2)]
data_class2 = data_class2.sample(frac=1) # shuffle to over-sampling
raw_data = pd.concat([data_class0, data_class1, data_class2, data_class2[:500]]) # Over-sampling

raw_data = raw_data.sample(frac=1) # shuffle df rows
X_comment= raw_data['comment']
Y_label = raw_data['new_label'].to_numpy()


def preprocess(X_comment, Y_label=None, for_training=False):
    '''
    Preprocess data.
    Input: X_comment: list of strings (comments)
           Y_label: list of labels (0,1,2). Required when for_training=True
           for_training: bool. If True: generate vocab and stuff
    Ouput: X_processed: tokenized and padded.
           Y_filter: list of labels filter according to X (only returned when for_training=True)
           vocab_X_size: size of vocab (only returned when for_training=True)
    '''

    # Delete all \n:
    # INFO: Mosses tokenizer (used below) can NOT deal with \n
    X_comment = [i.replace('\n',' ') for i in X_comment]

    # Convert to lowercase:
    X_comment = [i.lower() for i in X_comment]

    # Replace digits and punctuation by spaces:
    marks_to_del = '012345678'+string.punctuation
    table = str.maketrans(marks_to_del, ' '*len(marks_to_del))
    X_comment = [i.translate(table) for i in X_comment]

    # Remove repeated characters, eg., đẹppppppp
    X_comment = [re.sub(r'(.)\1+', r'\1', s) for s in X_comment] #Regex: https://docs.python.org/3/howto/regex.html#regex-howto 

    # [IMPORTANT] Convert charsets (bảng mã) TCVN3, VIQG... to Unicode
    X_comment = [unicodedata.normalize('NFC', text) for text in X_comment]

    # Tokenize text using Mosses tokenizer:
    # NOTE: Why choose Mosses tokenizer? See "How Much Does Tokenization Affect Neural Machine Translation?"    
    vi_tokenize = MosesTokenizer('vi')
    X_comment_tokenized = []
    X_comment_filtered = []
    Y_label_filtered = []
    for i in range(len(X_comment)): 
        comment = X_comment[i]
        tokens = vi_tokenize(comment) 
        if tokens!=[]: # since some sentences become empty after tokenization
            #!! Truncate sentences !!
            # NOTE: Beware! can strongly affect the performance.
            X_comment_tokenized.append(tokens[:N_WORDS_KEPT])            
            X_comment_filtered.append(comment) 
            if for_training:
                Y_label_filtered.append(Y_label[i])
    vi_tokenize.close()


    if for_training:
        joblib.dump(X_comment_tokenized, r'./datasets/X_comment_tokenized.joblib')
        joblib.dump(X_comment_filtered, r'./datasets/X_comment_filtered.joblib')
        joblib.dump(Y_label_filtered, r'./datasets/Y_label_filtered.joblib')
        print('\nDone making word lists.')

    if for_training:
        # Create vocabularies:
        words_list = [words for sentence in X_comment_tokenized for words in sentence]
        vocab, counts = np.unique(words_list, return_counts=True)
        vocab_count = {word:count for word, count in zip(vocab, counts)}
        print("full vocab.shape: ", vocab.shape)

        # Truncate the vocabulary (keep only words that appear at least min_occurrences times)
        truncated_vocab = dict(filter(lambda ele: ele[1]>=min_occurrences,vocab_count.items()))
        truncated_vocab = dict(sorted(truncated_vocab.items(), key=lambda item: item[1], reverse=True)) # Just to have low ids for most appeared words
        vocab_size = len(truncated_vocab)
        print("truncated vocal_size:", vocab_size)
        
        # Creat vocal list to convert words to ids:
        # NOTE: preserve 0, 1, 3 for end-of-seq, start-of-seq, and oov-word token
        vocab_list = ['<eos>', '<sos>', '<oov>']
        vocab_list.extend(list(truncated_vocab.keys()))
        joblib.dump(vocab_list,r'./datasets/vocab_list.joblib')  
        print('Done saving vocab_list.')   

        # Try encode, decoding some samples:
        temp_comment = X_comment_tokenized[:2]
        print('\ntemp_comment:',temp_comment)
        temp_encode = [list(map(lambda word: word_to_id(word, vocab_list), sentence)) for sentence in temp_comment]
        print('\ntemp_encode:',temp_encode)
    else:
        vocab_list = joblib.load(r'./datasets/vocab_list.joblib')        
        
    # Convert words (tokens) to ids: X_data: list of lists of token ids of X_comment_tokenized
    X_data = [list(map(lambda word: word_to_id(word, vocab_list), sentence)) for sentence in X_comment_tokenized]

    # Add end-of-seq and start-of-seq tokens:
    X_data =[[sos_id]+sentence+[eos_id] for sentence in X_data]
    
    # Pad zero to have all sentences of the same length (required when using batch_size>1):
    max_X_len = np.max([len(sentence) for sentence in X_data])
    X_padded = [sentence + [0]*(max_X_len - len(sentence)) for sentence in X_data]  
    
    if for_training:
        print('\nDONE loading and preprocessing data.')
        return X_padded, Y_label_filtered, vocab_list
    else: 
        return X_padded

X_padded, Y_label_filtered, vocab_list = preprocess(X_comment, Y_label, for_training=True)
vocab_X_size = len(vocab_list)
X = np.array(X_padded) 
Y = np.array(Y_label_filtered)    

#endregion


# In[3]: PART 3. TRAIN AN RNN MODEL
# Hyperparameters:
embed_size = 30 # NOTE: HYPERPARAM. embedding output size
n_units = 64 # NOTE: HYPERPARAM. Number of units in each layer. For simplicity, I have set the same number of units for all layers. However, feel free to change this if you wish (you can do that by finding where the variable n_units are in the code and change it one by one). 
n_epochs = 100 # NOTE: HYPERPARAM. Number of epochs to run training.
batch_size = 32 # NOTE: HYPERPARAM. batch_size
#region
# 3.1. Create model
model = keras.models.Sequential([
    keras.layers.Input(shape=[None]), #Input shape: [batch_size, n_steps]: both are varying hence None
    keras.layers.Embedding(vocab_X_size, embed_size, # convert word IDs into embeddings. Output shape: [batch_size, n_steps,embedding_size]: each word an embedding vector.
                           mask_zero=True), # mark the <pad> token. NOTE: MUST ensure padding token <pad> has id = 0
    keras.layers.GRU(n_units, return_sequences=True),
    keras.layers.GRU(n_units, return_sequences=True),
    keras.layers.GRU(n_units, return_sequences=True),
    keras.layers.GRU(n_units, return_sequences=True),
    keras.layers.GRU(n_units, return_sequences=False),
    keras.layers.Dense(3, activation="softmax") ])
model.summary()

#%% 3.2. Train the model
new_training = 0
if new_training:
    optimizer = 'nadam'
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    checkpoint_name = r'models/sentiment_GRU'+'_epoch{epoch:02d}_accuracy{accuracy:.4f}'+'.h5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='accuracy',save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='accuracy',patience=10,restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(r'logs/sentiment_train_log',embeddings_freq=1, embeddings_metadata='embed_file')
    
    history = model.fit(X, Y, epochs=n_epochs, batch_size=batch_size,
        callbacks = [model_checkpoint, early_stop, tensorboard] )
    #model.save(r'models/sentiment_GRU.h5')
    print('DONE training.')
else:
    print('NO new training.')

#endregion


# In[4]: PART 4. PREDICT
##### NOTE: specify correct model file name below: #####
model = keras.models.load_model(r'models/sentiment_GRU.h5')
#region
comment = ['sản phẩm mình nhận đúng như hình, chất lượng thì chắc dùng một thời gian mới biết.']
X_test_padded = preprocess(comment)
y_proba = model.predict(X_test_padded)

y_pred_label = np.argmax(y_proba)
label_meaning = {0: 'Không hài lòng', 1: 'Hài lòng', 2: 'Không rõ/Trung lập'}
print('\n', label_meaning[y_pred_label])
print('\nDetailed results:')
for key, value in label_meaning.items():
    print('  ',value,':', round(y_proba[0][key]*100,1),'%')

#endregion



