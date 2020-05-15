
import pandas as pd
import numpy as np
import seaborn as sns
from textblob import Word
from nltk.corpus import stopwords
import re
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import itemfreq
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,HashingVectorizer
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix




columns=['emotion','content']
data = pd.read_csv('./src/data/ISEAR.csv',names=columns)

print(data['content'][0])
print(data['content'].str.len())

data['content'] = data['content'].str.replace('\n', '')
print(data['content'][0])
print(data['content'].str.len())

# Replace full stop with blank
data['content'] = data['content'].str.replace('.', '')
print(data['content'][0])
print(data['content'].str.len())

#Remove irrelevant characters other than alphanumeric and space
data['content']=data['content'].str.replace('[^A-Za-z0-9\s]+', '')
print(data['content'][0])
print(data['content'].str.len())

#Remove links from the text
data['content']=data['content'].str.replace('http\S+|www.\S+', '', case=False)
print(data['content'][0])
print(data['content'].str.len())

#Convert everything to lowercase
data['content']=data['content'].str.lower()
print(data['content'][0])
print(data['content'].str.len())

#Removing Punctuation, Symbols
data['content'] = data['content'].str.replace('[^\w\s]',' ')
print(data['content'][0])
print(data['content'].str.len())


# Removing Stop Words using NLTK
nltk.download('stopwords')
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

print(data['content'][0])
print(data['content'].str.len())

#Lemmatisation

nltk.download('wordnet')
data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(data['content'][0])
print(data['content'].str.len())

#Correcting Letter Repetitions
def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

print(data['content'][0])
print(data['content'].str.len())

#Assign Target Variable
target=data.emotion
data = data.drop(['emotion'],axis=1)
print(target)
print(data)

#LabelEncoder for target
le=LabelEncoder()
target=le.fit_transform(target)
print(target)


# # **Split Data into train & test**
X_train, X_test, y_train, y_test = train_test_split(data,target,stratify=target,test_size=0.4, random_state=42)
print(X_train)
print(y_train)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.content)
X_test_counts =count_vect.transform(X_test.content)
print('Shape of Term Frequency Matrix: ',X_train_counts.shape)




# First experiment 


# Naive Bayes Model
clf = MultinomialNB().fit(X_train_counts,y_train)
predicted = clf.predict(X_test_counts)
nb_clf_accuracy = np.mean(predicted == y_test) * 100
print(nb_clf_accuracy)

experiment_id = mlflow.create_experiment(name='Naive Bayes')
with mlflow.start_run(experiment_id=experiment_id):

    mlflow.log_metric("accuracy",nb_clf_accuracy)
    mlflow.sklearn.load_model(clf,'model')
    modelpath='./src/models'('Naive Bayes',10)
    mlflow.sklearn.save_model(clf,modelpath)
    mlflow.log_artifact('first.png')



