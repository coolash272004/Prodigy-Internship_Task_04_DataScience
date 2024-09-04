import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix , classification_report ,accuracy_score,ConfusionMatrixDisplay

name_column = ['id', 'entity', 'target', 'Tweet content']
datafile2 = pd.read_csv('F:/Prodigy_Projects/Task4/dataset/twitter_training.csv', names = name_column)
print(datafile2)
print(datafile2.info())
df01 = datafile2.drop(columns=['id','entity'],axis=1)
print(df01)
print(df01.isna().sum())
print(df01.dropna(inplace = True))
count = df01['target'].value_counts()
print(count)

#-----------------------        Plotting a Bargraph     -----------------------#
plt.figure(figsize=(12, 8))
sns.barplot(x=count.index, y=count.values, palette='viridis')
plt.title('target_counts')
plt.xlabel('target')
plt.ylabel('count')
plt.show()

#-----------------------------     Preprocessing     --------------------------#
ps = PorterStemmer()
stops = set(stopwords.words('english'))
def preprocessing_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    token=text.split()
    token=[ps.stem(word) for word in token if word not in stops]
    return ' '.join(token)

df01['Tweet content']=df01['Tweet content'].apply(preprocessing_text)
df01['Tweet content']

#------------------------      Feature Extraction      ------------------------#
tf=TfidfVectorizer(max_features=5000)
x=tf.fit_transform(df01['Tweet content'])
y=df01['target']
print(x.shape)
print(y.shape)

#---------------------------    Data Splitting       --------------------------#
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=42)


#---------------------------    Model Building       --------------------------#
models = {
    'Naive Bayes': MultinomialNB(),
    'Decision tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n--- {name} ---\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)