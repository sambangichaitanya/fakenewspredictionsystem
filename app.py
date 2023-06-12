import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

app = Flask(__name__)
model = pickle.load(open("model.pkl",'rb'))
news_data = pd.read_csv('train.csv')
news_data = news_data.fillna('')
news_data['content'] = news_data['author'] + news_data['title']
x1 = news_data.drop(columns='label',axis=1)
y1 = news_data['label']
news_data['content'] = news_data['content'].apply(stemming)

x1 = news_data['content'].values
vectorizer.fit(x1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    d1 = request.form.get('a')
    d2 = request.form.get('b')
    x = pd.DataFrame({'string':[d1+d2]})
    x = x['string'].apply(stemming)
    x = x.values
    x = vectorizer.transform(x)
    result = model.predict(x)
    if result[0]==1:
       return render_template('index.html',predicted_text="The news is fake")
    else:
       return render_template('index.html',predicted_text="The news is real")

if __name__=='__main__':
    app.run(debug=True)
