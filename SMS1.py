import pandas as pd
text_messages = pd.read_csv(r"C:\Users\anind\OneDrive\Desktop\spam.csv", encoding = 'ISO-8859-1', usecols = ['v1', 'v2'])


print(text_messages)

import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

text_messages['v2'] = text_messages['v2'].apply(preprocess_text)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(text_messages['v2']).toarray()

print(X)



y = pd.get_dummies(text_messages['v1'])
y = y.iloc[:,1].values

print(y)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detection_model = MultinomialNB().fit(X_train, y_train)


''''''
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

y_pred = spam_detection_model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)


import matplotlib.pyplot as plt

labels = [0,1]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()


'''
new_messages = ['Get a free cruise now!', 'Hey, can you pick up some milk on your way home?']
new_messages = [preprocess_text(msg) for msg in new_messages]
new_messages = cv.transform(new_messages)
predictions = spam_detection_model.predict(new_messages)
print('Predictions:', predictions)

'''