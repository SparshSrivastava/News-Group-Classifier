# Importing all the stopwords into a dictionary named stopwords from the text file StopWords
stopwords={}
with open('StopWords.txt','r') as fp:
    for x in fp.read().split('\n'):
        stopwords[x]=1

# Defining a function ExtractWords to extract the words from the folders of mini_newsgroups
def ExtractWords(data):
    arr=[]
    py=data.split(' ')
    for x in range(len(py)):
        word=''
        word=py[x].replace('\n','')
        word=word.replace('\r','')
        word=word.replace('\t','')
        word=word.replace('(','')
        word=word.replace(')','')
        word=word.replace(').','')
        word=word.replace('.','')
        word=word.replace('"','')
        word=word.replace(",",'')
        word=word.lower()
        if word=='':
            continue
        elif word in stopwords.keys():
            continue
        else:
            arr.append(word)
    return arr

# Importing the data from the folders
path='mini_newsgroups'
data=[]
Y=[]

import os

for item_names in os.listdir(path):
    item_names=os.path.join(path,item_names)
    print(item_names)
    for MyPath in os.listdir(item_names):
        MyPath=os.path.join(item_names,MyPath)
        arr=[]
        with open(MyPath,'r') as fp:
            arr=ExtractWords(fp.read())
        data.append(arr)
        Y.append(item_names)

# Removing the mini_newsgroups/ from the Y therefore defining a function ExtractLabel to do so
def ExtractLabel(labels):
    for x in range(len(labels)):
        labels[x]=labels[x].replace('mini_newsgroups/','')
    return labels

Y=ExtractLabel(Y)


#Creating a dictionary to store the frequencies of words in the given data with key being the words and their freqyuency being the value
dictionary={}
for word_list in data:
    for words in word_list:
        dictionary[words]=dictionary.get(words,0)+1
        
#Now removing those words having frequency less than some value(threshold) in this model taken 1
for key in dictionary.keys():
    if(dictionary[key]==1):
        del(dictionary[key])

# Now Calculating the number of features that are useful for the creation of X matrix
num_features=len(dictionary.keys())

# Making a list of all words in the list called words_dictionary
words_dictionary=dictionary.keys()

#Creating the 2D vector X
import numpy as np
# np.mgrid(size_row,size_column) creates 2 copies so need to use NoUse array
X,NoUse=np.mgrid[0:len(data),0:num_features]

for i in range(len(data)):
    #We need another dictionary to maintain the frequencies of words in a particular row of data
    d={}
    for j in data[i]:
        d[j]=d.get(j,0)+1
    for j in range(num_features):
        X[i][j]=d.get(words_dictionary[j],0)


#Splitting the dataset into training and testing data
from sklearn import model_selection
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,random_state=1)

# Predicting the class using Naive Bayes Multinomial classifier
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()

#Training the model against the training set
mnb=mnb.fit(X_train,Y_train)

#Predicting the values against the test data
Y_pred=mnb.predict(X_test)

#Printing the Confusion Matrix and Classification Report to see the efficiency of the model
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))

# Predicting the class using the Naive Bayes Gaussian Classifier
from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()

gb.fit(X_train,Y_train)

Y_pred_gb=gb.predict(X_test)

print(classification_report(Y_test,Y_pred_gb))
print(confusion_matrix(Y_test,Y_pred_gb))

#Predicting the class using the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()

forest.fit(X_train,Y_train)

Y_pred_forest=forest.predict(X_test)

print(classification_report(Y_test,Y_pred_forest))
print(confusion_matrix(Y_test,Y_pred_forest))

#Predicting the class using the Logistic Regression
from sklearn.linear_model import LogisticRegression
clg=LogisticRegression()

clg.fit(X_train,Y_train)

Y_pred_Logistic=clg.predict(X_test)

print(classification_report(Y_test,Y_pred_Logistic))
print(confusion_matrix(Y_test,Y_pred_Logistic))

#Conclusion:
# The Logistic Regression is doing preferably well on this dataset followed by Naive Bayes Multinomial Classifier, then Random Forest Classifier and then Naive Bayes Gaussian Classifier as evident by the Classification Report