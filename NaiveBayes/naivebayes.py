import pandas as pd

dataset = './dataset/SMSSpamCollection'

##### データの読み込み #####

#read_csv: csvファイル読み込み
#read_table: tsvファイル読み込み
df = pd.read_table(dataset, sep='\t', header=None, names=['label', 'sms_message'])
#print(df.head(3))


##### ラベルを0,1に #####

#df.ix[:,0] = [1 if label=='spam' else 0 for label in df.ix[:,0]]
#print(df.ix[:,0])
#print(df.head(3))

# 模範回答
#df['label'] = df.label.map({'ham':0, 'spam':1})
#print(df.head(3))


"""
##### Bag of Words from scratch #####

# 1. 全ての文字を小文字(lowercase)に変換
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']
lowercase_doc = []
for doc in documents:
    lowercase_doc.append(doc.lower())  #stringを小文字化するメソッド lower()
#print(lowercase_doc)

# 2. 句読点(punctuation)を除く
import string
removal_punc_doc = []
for ldoc in lowercase_doc:
    removal_punc_doc.append(ldoc.translate(str.maketrans('','', string.punctuation)))
#print(removal_punc_doc)

# 3. Tokenization(文字ごとに分割)
preprocessed_doc = []
for redoc in removal_punc_doc:
    preprocessed_doc.append(redoc.split(' '))  #半角スペースで分割
#print(preprocessed_doc)

# 4. Count frequency
from collections import Counter
frequency_list = []
bow = Counter('')
for word in preprocessed_doc:
    frequency_counts = Counter(word)
    frequency_list.append(frequency_counts)
    bow += frequency_counts  #Counterは通常の辞書型にない「＋」がサポートされている
#print(frequency_list)
print(bow)


##辞書を結合(単に書いてみただけ)
#bow = []
#for num, freq in enumerate(frequency_list):
#    if(num==0): bow = freq
#    else:
#        bow.update(freq)
#print(bow)


"""



"""
##### Bag of Words from sklearn #####

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

from sklearn.feature_extraction.text import CountVectorizer

#デフォルトでlowercase(小文字)，punctuation(句読点)は除く仕様
count_vector = CountVectorizer()
count_vector.fit(documents)
#print(count_vector.get_feature_names())  #abc順に単語並ぶ

doc_array = count_vector.transform(documents).toarray()
#print(doc_array)

#データフレームで見やすく
frequency_mat = pd.DataFrame(doc_array,
                            columns = count_vector.get_feature_names())
print(frequency_mat)

"""



##### Training and testing sets #####

# 1. 訓練データとテストデータに分割
from sklearn.cross_validation import train_test_split

#デフォルトは，4:1（訓練：テスト）にランダム分割
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)
#print('Number of rows in the total set: {}'.format(df.shape[0]))
#print('Number of rows in the training set: {}'.format(X_train.shape[0]))
#print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# 2. データをBag of Wordsに変換

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(lowercase=True)
training_data = count_vector.fit_transform(X_train).toarray()
test_data = count_vector.transform(X_test).toarray()
#print(count_vector.get_feature_names())  #abc順に単語並ぶ
#print(traing_data)
#print(test_data)

l1 = [1,1,1,1,1]
l2 = [0,0,0,0,1]
import numpy as np
print(np.sum([1 if i==j else 0 for i, j in zip(l1, l2)]))
