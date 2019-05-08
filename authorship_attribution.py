import os
import json
import codecs
import operator
import re
import string
import argparse
import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer,PunktSentenceTokenizer, TreebankWordTokenizer
from nltk.corpus import stopwords, webtext
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import scale
from sklearn import utils
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC ,SVC
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

punk_sent_tokenizer = PunktSentenceTokenizer(webtext.raw('overheard.txt'))
vader = SentimentIntensityAnalyzer()

def Read_json(path):
    with codecs.open( path , 'r' , encoding="utf-8") as f:
        return json.load(f)

def Save_json(path , data):
    with codecs.open(path , 'w',encoding='utf-8') as J:
        json.dump(data, J , indent=4)

def Read_text(path):
    with codecs.open( path , 'r' , encoding='utf-8') as f:
        return f.read()

def Mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def tfidf_Preprocessing(text , _stopwords):
    stemmer = PorterStemmer()
    text = text.replace('-' , '')
    text = text.replace('.' , '')
    text = text.replace('”' , '')
    text = text.replace('’' , '')
    text = text.replace('“' , '')
    text = text.replace('‘' , '')
    text = text.replace('–','')
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([stemmer.stem(word).lower() for word in WordPunctTokenizer().tokenize(nopunc) 
                    if stemmer.stem(word).lower() not in _stopwords])

def tfidf_represent_text(text ):
    tokens = WordPunctTokenizer().tokenize(text)
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency

def tfidf_extract_vocabulary(texts , ft):
    occurrences=defaultdict(int)
    for text in texts:
        text_occurrences=tfidf_represent_text(text)
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram]+=text_occurrences[ngram]
            else:
                occurrences[ngram]=text_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i]>=ft:
            vocabulary.append(i)
    return vocabulary

def ngram_represent_text(text,n):
    if n>0:
        tokens = [text[i:i+n] for i in range(len(text)-n+1)]
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency

def ngram_extract_vocabulary(texts,n,ft):
    occurrences=defaultdict(int)
    for text in texts:
        text_occurrences=ngram_represent_text(text,n)
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram]+=text_occurrences[ngram]
            else:
                occurrences[ngram]=text_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i]>=ft:
            vocabulary.append(i)
    return vocabulary

def buildWordVector(imdb_w2v, text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def identify_authors( input_dir , out_dir , pt , n , ft ):
    max_abs_scaler = preprocessing.MaxAbsScaler()
    Mkdir(out_dir)
    stopwords_list = {'en': set(stopwords.words('english')) , 'fr':set(stopwords.words('french')),
                      'sp': set(stopwords.words('spanish')) , 'it':set(stopwords.words('italian'))}
    problems = Read_json(os.path.join( input_dir , "collection-info.json"))
    for problem in problems:
        print("Working on Problem :::" , problem['problem-name'])
        print("\t language: " ,problem['language'])
        
        problem_path = os.path.join(input_dir ,problem['problem-name'])
        problem_info = Read_json(os.path.join(problem_path, 'problem-info.json'))        
        candidates = [candidate['author-name'] for candidate in problem_info['candidate-authors']]
        train_set , train_labels = [], []
        for candidate in candidates:
            candidate_dir = os.path.join(problem_path , candidate)
            for text in os.listdir(candidate_dir):
                train_set.append(Read_text(os.path.join(candidate_dir ,text)))
                train_labels.append(candidate)
        unknowns_dir = os.path.join(problem_path , "unknown")
        test_set , unks = [] , []
        for unk in os.listdir(unknowns_dir):
            test_set.append(Read_text(os.path.join(unknowns_dir , unk)))
            unks.append(unk)

        tfidf_train_set = [tfidf_Preprocessing(text , stopwords_list[problem['language']]) 
                        for text in train_set]
        tfidf_test_set = [tfidf_Preprocessing(text , stopwords_list[problem['language']])
                        for text in test_set]
                
        word2vec_train_set = [text.split() for text in tfidf_train_set]
        word2vec_test_set = [text.split() for text in tfidf_test_set]
        n_dim = 300
        word2vec_model = Word2Vec(sg=1, size=n_dim, min_count=1, workers=7)
        word2vec_model.build_vocab(word2vec_train_set)
        for epoch in range(20):
            word2vec_model.train(word2vec_train_set ,total_examples=word2vec_model.corpus_count, epochs=5)
        for epoch in range(20):
            word2vec_model.train(word2vec_test_set ,total_examples=word2vec_model.corpus_count, epochs=5)
        word2vec_train = np.concatenate([buildWordVector(word2vec_model, text , n_dim) for text in word2vec_train_set])
        word2vec_train = scale(word2vec_train)
        word2vec_test = np.concatenate([buildWordVector(word2vec_model, text , n_dim) for text in word2vec_test_set])
        word2vec_test = scale(word2vec_test)
        word2vec_scaled_train_data = max_abs_scaler.fit_transform(word2vec_train)
        word2vec_scaled_test_data = max_abs_scaler.transform(word2vec_test)
        word2vec_clf = CalibratedClassifierCV(OneVsRestClassifier(LogisticRegression(C=0.01)))
        word2vec_clf.fit(word2vec_scaled_train_data, train_labels)
        word2vec_predictions = word2vec_clf.predict(word2vec_scaled_test_data)
        word2vec_proba = word2vec_clf.predict_proba(word2vec_scaled_test_data)
        
        tfidf_vocab = tfidf_extract_vocabulary(tfidf_train_set , ft )
        tfidf_vectorizer = TfidfVectorizer(vocabulary=tfidf_vocab, norm=None, strip_accents=False)
        tfidf_train_data = tfidf_vectorizer.fit_transform(tfidf_train_set)
        tfidf_test_data = tfidf_vectorizer.fit_transform(tfidf_test_set)
        tfidf_scaled_train_data = max_abs_scaler.fit_transform(tfidf_train_data)
        tfidf_scaled_test_data = max_abs_scaler.transform(tfidf_test_data)
        tfidf_clf = CalibratedClassifierCV(OneVsRestClassifier(LinearSVC(C=0.01)))
        tfidf_clf.fit(tfidf_scaled_train_data, train_labels)
        tfidf_predictions = tfidf_clf.predict(tfidf_scaled_test_data)
        tfidf_proba = tfidf_clf.predict_proba(tfidf_scaled_test_data)
        
        ngram_vocabulary = ngram_extract_vocabulary(train_set , n , ft)
        ngram_vectorizer = CountVectorizer(strip_accents=False, analyzer='char',ngram_range=(n,n),lowercase=False,vocabulary=ngram_vocabulary)  
        ngram_train_data = ngram_vectorizer.fit_transform(train_set)
        ngram_train_data = ngram_train_data.astype(float)
        for i in range(len(train_set)):
            ngram_train_data[i]=ngram_train_data[i]/len(train_set[i])
        ngram_test_data = ngram_vectorizer.transform(test_set)
        ngram_test_data = ngram_test_data.astype(float)
        for i in range(len(test_set)):
            ngram_test_data[i] = ngram_test_data[i]/len(test_set[i])
        ngram_scaled_train_data = max_abs_scaler.fit_transform(ngram_train_data)
        ngram_scaled_test_data = max_abs_scaler.transform(ngram_test_data)
        ngram_clf = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=0.01 , kernel='linear')))
        ngram_clf.fit(ngram_scaled_train_data, train_labels)
        ngram_predictions = ngram_clf.predict(ngram_scaled_test_data)
        ngram_proba = ngram_clf.predict_proba(ngram_scaled_test_data)

        proba = []
        predictions = []
        for i in range(0,len(test_set)):
            proba.append((word2vec_proba[i] + ngram_proba[i] + tfidf_proba[i])/3)
            predictions.append(candidates[np.argmax(proba[i])])
        count = 0
        for i in range(0,len(predictions)):
            sproba = sorted(proba[i],reverse=True)
            if sproba[0]-sproba[1] < pt:
                predictions[i] = u'<UNK>'
                count = count + 1
        print('\t',count ,'texts left unattributed')
        out_data =[{'unknown-text':unks[i],'predicted-author': predictions[i]} for i in range(len(test_set))]
        Save_json(os.path.join(out_dir , 'answers-'+problem['problem-name']+'.json') , out_data)
        print('\t answers saved to file answers-' + problem['problem-name'] + '.json')
        print("----------------------------------------------------------------")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to input dataset")
    parser.add_argument('-o', '--output', help="path to output directory")
    parser.add_argument('-n', help="n gram", default=4)
    parser.add_argument('-ft', help="frequency term for tfidf and ngram", default=5)
    parser.add_argument('-pt', help="threshold for UNK authors", default=0.08)
    args = parser.parse_args()
    if args.input is None or args.output is None:
        parser.print_usage()
        exit()
    return args

if __name__=="__main__":
    args = get_args()
    identify_authors( input_dir = args.input , out_dir = args.output , pt = args.pt ,n = args.n , ft = args.ft)
