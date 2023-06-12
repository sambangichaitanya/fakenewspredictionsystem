import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import nltk
nltk.download('stopwords')

news_data = pd.read_csv('train.csv')
news_data = news_data.fillna('')
news_data['content'] = news_data['author'] + news_data['title']
x = news_data.drop(columns='label',axis=1)
y = news_data['label']


port_stem = PorterStemmer()


def stemming(content):
  stemmed_content = re.sub('[^a-zA-z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

news_data['content'] = news_data['content'].apply(stemming)


x = news_data['content'].values
y = news_data['label'].values


vectorizer = TfidfVectorizer()

vectorizer.fit(x)

x = vectorizer.transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

model = LogisticRegression()
model.fit(x_train,y_train)


pickle.dump(model,open("model.pkl",'wb'))
mod1 = pickle.load(open("model.pkl",'rb'))
print(mod1.predict(x_test))
print(x_test)
