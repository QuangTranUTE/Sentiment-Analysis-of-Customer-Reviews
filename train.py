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
from numpy.core.defchararray import count
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
from sklearn.model_selection import train_test_split

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
    # INFO: Mosses tokenizer (used below) reserves punctuation (what we want).
    #       but its can NOT deal with \n
    X_comment = [i.replace('\n',' ') for i in X_comment]

    # Convert to lowercase:
    X_comment = [i.lower() for i in X_comment]

    # Replace digits and punctuation by spaces (to remove them):
    marks_to_del = '012345678'+string.punctuation
    table = str.maketrans(marks_to_del, ' '*len(marks_to_del))
    X_comment = [i.translate(table) for i in X_comment]

    # Remove repeated characters, eg., đẹppppppp
    # tryex = re.sub(r'(.)\1+', r'\1', 'san  phẩmmmmm   loooiiiii :)))))))))') 
    X_comment = [re.sub(r'(.)\1+', r'\1', s) for s in X_comment] #Regex: https://docs.python.org/3/howto/regex.html#regex-howto 

    # Add accent (tool does not work so well, so DON'T use!)
    # from pyvi import ViUtils
    # ViUtils.add_accents('san phẩm chất luong') 

    # [IMPORTANT] Convert charsets (bảng mã) TCVN3, VIQG... to Unicode
    X_comment = [unicodedata.normalize('NFC', text) for text in X_comment]

    # [not really necessary] Connect words (in VNese, eg., "cực kỳ" is 1 word, "sản phẩm" is 1 word)
    #from pyvi import ViTokenizer
    #X_comment = [ViTokenizer.tokenize(i) for i in X_comment]

    print('\nSome processed comments:', X_comment[:10])

    # Tokenize text using Mosses tokenizer:
    # NOTE: Why choose Mosses tokenizer? See "How Much Does Tokenization Affect Neural Machine Translation?"    
    vi_tokenize = MosesTokenizer('vi')
    X_comment_tokenized = []
    X_comment_filtered = []
    Y_label_filtered = []
    for i in range(len(X_comment)): 
        comment = X_comment[i]
        tokens = vi_tokenize(comment) 
        #tokens = comment.split() # may try this instead of MosesTokenizer
        if tokens!=[]: # since some sentences become empty after tokenization
            #!! Truncate sentences !!
            # NOTE: Beware! can strongly affect the performance.
            X_comment_tokenized.append(tokens[:N_WORDS_KEPT])            
            X_comment_filtered.append(comment) 
            if for_training:
                Y_label_filtered.append(Y_label[i])
    vi_tokenize.close()


    if for_training:
        # Have a look at a comment and its tokens
        comment = 'Hàng xài ok nha mn,quên chụp rồi,giao_hàng nhanh_chóng đầy đủ ❤️❤️😂😂😂😂'
        print('\n',comment)
        with MosesTokenizer('vi') as vi_tokenize:
            tokens = vi_tokenize(comment)
        print('\n',tokens)
        with MosesDetokenizer('vi') as detokenize:
            comment_back = detokenize(tokens)
        print('\n',comment_back)

        # n_samples = len(X_comment_filtered)
        # joblib.dump(n_samples, r'./datasets/n_samples.joblib')
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
        #joblib.dump(vocab_size,r'./datasets/vocab_size.joblib')

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
        #vocab_size = joblib.load(r'./datasets/vocab_size.joblib')

    # Convert words (tokens) to ids: X_data: list of lists of token ids of X_comment_tokenized
    X_data = [list(map(lambda word: word_to_id(word, vocab_list), sentence)) for sentence in X_comment_tokenized]

    # Add end-of-seq and start-of-seq tokens:
    X_data =[[sos_id]+sentence+[eos_id] for sentence in X_data]
    
    # Pad zero to have all sentences of the same length (required when using batch_size>1):
    max_X_len = np.max([len(sentence) for sentence in X_data])
    X_padded = [sentence + [0]*(max_X_len - len(sentence)) for sentence in X_data]  
    
    # vocab_X_size = vocab_size + 3
    
    if for_training:
        print('\nDONE preprocessing data.')
        return np.array(X_padded), np.array(Y_label_filtered), vocab_list
    else: 
        return np.array(X_padded)

# Hyperparameters for preprocessing data:
N_WORDS_KEPT = 150 # NOTE: HYPERPARAM. Number of words to keep in each sample (a line in txt files)             
min_occurrences = 4 # NOTE: HYPERPARAM. Each word appear many times in the dataset. We only keep the words that occur > min_occurrences in the dataset. Amitabha 

#endregion


# In[2]: PART 2. LOAD AND PREPROCESS DATA
# Hyperparameters:
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
# [NOTE: HYPERPARAM] Under-sampling and over-sampling classes to reduce data imbalance:
data_class0 = raw_data[(raw_data.label==0)]; print(f'Length of data_class0: {len(data_class0)} reviews.')
data_class1 = raw_data[(raw_data.label==1)]; print(f'Length of data_class1: {len(data_class1)} reviews.')
data_class2 = raw_data[(raw_data.label==2)]; print(f'Length of data_class2: {len(data_class2)} reviews.')
# Resample:
data_class1 = data_class1[:30000] # Under-sampling class 1
data_class2 = data_class2.sample(frac=1) # Shuffle to do over-sampling below
raw_data = pd.concat([data_class0, data_class1, data_class2, data_class2[:400]]) # Over-sampling class 2
print('\nAFTER RESAMPLING:')
print(f'Length of data_class0: {len(data_class0)} reviews.')
print(f'Length of data_class1: {len(data_class1)} reviews.')
print(f'Length of data_class2: {len(data_class2)+400} reviews.')

# Shuffle dataframe rows:
raw_data = raw_data.sample(frac=1) 

# Split comment and label AND convert to proper data types
X_comment= raw_data['comment'].to_numpy(dtype=np.str)
Y_label = raw_data['label'].to_numpy(dtype=np.int8)

load_processed_data = False
if not load_processed_data:
    X_processed, Y_processed, vocab_list = preprocess(X_comment, Y_label, for_training=True)
    vocab_X_size = len(vocab_list)
    X, X_test, Y, Y_test = train_test_split(X_processed, Y_processed, test_size=0.05, random_state=48)
    joblib.dump(X, r'datasets/X.joblib')
    joblib.dump(Y, r'datasets/Y.joblib')
    joblib.dump(X_test, r'datasets/X_test.joblib')
    joblib.dump(Y_test, r'datasets/Y_test.joblib')
    joblib.dump(vocab_list, r'models/vocab_list.joblib')
else:
    X = joblib.load(r'datasets/X.joblib')
    Y = joblib.load(r'datasets/Y.joblib')
    X_test = joblib.load(r'datasets/X_test.joblib')
    Y_test = joblib.load(r'datasets/Y_test.joblib')
    vocab_list = joblib.load(r'models/vocab_list.joblib')
    vocab_X_size = len(vocab_list)
    print('\nDone LOADING processed data.')
#endregion



# In[3]: PART 3. TRAIN AN RNN MODEL
# Hyperparameters:
embed_size = 30 # NOTE: HYPERPARAM. embedding output size
n_units = 64 # NOTE: HYPERPARAM. Number of units in each layer. For simplicity, I have set the same number of units for all layers. However, feel free to change this if you wish (you can do that by finding where the variable n_units are in the code and change it one by one). 
n_epochs = 50 # NOTE: HYPERPARAM. Number of epochs to run training.
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
    keras.layers.GRU(n_units, return_sequences=False),
    keras.layers.Dense(3, activation="softmax") ])
model.summary()

#%% 3.2. Train the model
new_training = True
if new_training:
    optimizer = 'nadam'
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    checkpoint_name = r'models/sentiment_GRU'+'_epoch{epoch:02d}_val_accuracy{val_accuracy:.4f}'+'.h5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_accuracy',save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=10,restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(r'logs/sentiment_train_log',embeddings_freq=1, embeddings_metadata='embed_file')
    
    model.fit(X, Y, epochs=n_epochs, batch_size=batch_size,
        callbacks = [model_checkpoint, early_stop, tensorboard],
        validation_data=(X_test, Y_test) )
    #model.save(r'models/sentiment_GRU.h5')
    print('DONE training.')
else:
    print('NO new training.')

#endregion


# In[4]: PART 4. PREDICT
##### NOTE: specify correct model file name below: #####
model = keras.models.load_model(r'models/sentiment_GRU_epoch30_accuracy0.9568.h5') # BEST model here
##### IMPORTANT NOTE: MUST load the RIGHT vocab_list file that goes with the trained model used.
vocab_list = joblib.load(r'datasets/vocab_list.joblib')
label_meaning = {0: 'Không hài lòng', 1: 'Hài lòng', 2: 'Không rõ/Trung lập'}
#region
# 4.1. Predict 1 comment
comment = ['mẫu mã chất liệu rất ưng ý']
X_test_padded = preprocess(comment)
y_proba = model.predict(X_test_padded)
y_pred_label = np.argmax(y_proba[0])
print('\n', label_meaning[y_pred_label])
print('\nDetailed results:')
for key, value in label_meaning.items():
    print('  ',value,':', round(y_proba[0][key]*100,1),'%')

#%% 4.2. Predict a bunch of comments
comments = ['sp như hình giao hàng chậm', 
            'Dây lưng rất đẹp, mặt khóa trẻ trung, dễ phối đồ. sẽ tiếp tục ủng hộ shop',
            'day dep nhe. nen mua m. a',
            'nón y như trong hình nha , ko có lỗi j , nói chung là rất ok lun , aii mún mua thì mua ik , mua xong nhớ cho nhận xét đề người sau mua nữa nha,',
            'Giay dep mjnh rat hai long.se ung ho shop nhieu nua,',
            'Chất lượng ổn nha mọi ngừoi',
            'shop bán hàng đẹp ok.cho shop 5 sao.nhân viên giao hàng giao tiếp rất tốt lịch sự',
            'Với giá như vậy là quá tốt rồi Giao hàng cực nhanh. Balo xịn xò giá tốt',
            'rất ưng cái bụng cho sản phẩm của các bạn, chất lượng tốt giá thành ok, nói chung là tốt',
            'Đã nhận hàng. Đóng gói cẩn thận, giày nhẹ, đi thử thì thấy êm, hài lòng!', 
            'chưa thực sự giống hình nhưng giao trước dự kiến đóng gói đẹp shipper thân thiện nhiệt tình 5sao',
            'Shop gửi nhầm mình cái đồng hồ',
            'Tất mềm mịn, chất hơi mỏng, qua mắt cá chân. Sản phẩm ổn so với giá, nên mua',
            'mặt đồng hồ lồi, hơi ko được mỏng như mình mong, nhưng mà vẫn đẹp vì màu đen',
            'Chưa sd nhưng nhìn qua thì thấy balo chắc chắn',
            'Lúc đầu cứ lo là sản phẩm bên shop không được như mong muốn vì giá quá mềmhôm nay em vừa nhận cặp xong và kiếm cái để chê nhưng thật sự không có ạ',
            'áo hơi rộng so với size',
            'Va li của mình bị nứt, shop ko kiểm tra hàng trước khi gửi',
            'Hơi giãn',
            'Khẩu trang hơi mỏng mỗi 1 lớp. Có ra 2 lớp sẽ đẹp hơn và dùng thích hơn',
            'sp hok bền như mong muốn.Mới mua về sau 1 ngày bị bong tróc màu rồi bị sau lớp da nữa.Thất vọng.',
            'Mũ hơi rộng với gần lưởi mủ 2 bên có cái j chọt ra , khi mang vào ở phần trên bó lấy trán nên hơi đau',
            'Nhận hàng hơi bị thất vọng, số lượng hạt xấu ( trầy , tróc, ố màu) quá nhiều. Mua làm quà tặng mà vòng thế này không dám đem tặng luôn! (-.-)',
            'Hơi thất vọng vì giao lộn hình xương rồng',
            'Áo không đc bền chắc cho lắm, tiền nào của nấy.',
            'Quá thất vọng. Mua mới mà dây đeo túi rỉ sắt. Xem xong ko biết mình mua túi để lâu rồi thì phải🙂',
            'Mở balo ra thì hàng bị lỗi, không kiểm tra hàng trước khi giao hàng',
            'Shop giao hàng mà không kiểm tra kỹ , túi xách giao mà không có dây đeo vai vậy shop?Vui lòng gừi bổ sung giúp.',
            'Tôi đặt màu xám đen như hình minh họa shop gửi hàng về tôi nhận lại như này đây Làm ăn thiếu uy tín quá',
            'Không có dây sạc mà ko lung linh như hình',
            'Chất liệu ko ổn lắm, nhiều vinilong, ko như giới thiệu',
            'sản phản ko như mong muốn',
            'Rất không hài lòng với sản phẩm',
            'Mới vừa nhận hàng thì đồng hồ k có pin, k chạy được. Dây thì đứt ra 1 bên. Cho hỏi cái shop này làm ăn kiểu gì vậy ạ??? Giá rẻ thật nhưng làm ơn đừng làm mất thời gian của khách hàng như vậy.',
            'hôm bữa mình mua một cặp . những chỉ gío một cái shop làm ăn ko uy tín... tôi ko thích kiểu phục vụ này... với lại tôi mua màu trắng lại vẩn chuyển cho màu đen',
            'E chọn mua mặt xanh size nữ mà lại ship cho e là mặt trắng size nam ạ',
            'Mua về giặt thì ra màu quá trời luôn:( đang khô nên không có hình chụp',
            'sản phẩm mình nhận đúng như hình, nhưng bao bì của sản phẩm đóng gói là hàng Việt Nam không phải như vậy. sản phẩm này của Trung Quốc',
            'tạm ổn hơi dính keo một chút trên dép, nhưng nhìn có vẻ chắc chắn']
X_test_padded = preprocess(comments)
y_proba = np.round(model.predict(X_test_padded),3)
y_pred_label = np.argmax(y_proba, axis=1)
print('\n\nResults (in the same order of comments:\n')
for y, comment in zip(y_pred_label, comments):
    print(label_meaning[y], ':', comment[:50])

# Compute avg. rating:
y_values, counts = np.unique(y_pred_label, return_counts=True) 
avg_rating = (counts[0]*1 + counts[1]*5 + counts[2]*2.5)/len(comments)
print('\nAvg. rating of these reviews (1-5 stars):', round(avg_rating,1))

#endregion

