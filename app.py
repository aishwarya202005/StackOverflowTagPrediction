#for deployment download the pkl files from https://drive.google.com/drive/u/1/folders/1VzeyVZnPfX6NLkr2nIGIkvS5ni_fMLzN


from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
from nltk.tokenize import word_tokenize
nltk.download("punkt")
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib





import flask
app = Flask(__name__)


###################################################
def striphtml(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(data))
    return cleantext
def cleaner(title,body):  # define tweet_cleaner function to clean the tweets
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    body=re.sub('<code>(.*?)</code>', '',body, flags=re.MULTILINE|re.DOTALL)
    body=striphtml(body.encode('utf-8'))
    title=title.encode('utf-8')
    body=str(title)+" "+str(title)+" "+str(title)+" "+str(body)
    body=re.sub(r'[^A-Za-z]+',' ',body)
    words=word_tokenize(str(body.lower()))
    body=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))
    return body

###################################################


@app.route('/')
def hello_world():
    return 'Welcome!!!! Go to index page to perform Auotmatic tag generation'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('model.pkl')
    tf_vect = joblib.load('tfidf_text.pkl')
    features=joblib.load('features.pkl')
    to_predict_list = request.form.to_dict()

    clean_text = cleaner(to_predict_list['title'],to_predict_list['description'])


    pred = clf.predict(tf_vect.transform([clean_text]))
    tag=''
    for i in range(len(features)):
        if pred[0,i]==1:
            tag=tag + " "+ str(features[i])
    
    return jsonify({'prediction': tag})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
