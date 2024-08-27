#-----------------------------------------------------#
#                  ANALYSE & VISUALIZE I
#-----------------------------------------------------#
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

col_names = ['ID', 'Entity', 'Sentiment', 'Content']
df = pd.read_csv('C:/Users/ABC/Desktop/Prodigy Internship/Data Science/twitter_training.csv', names=col_names)

# Display initial data information
print(df.head())
print(df.shape)
print(df.describe)
print(df.isnull().sum())
print(df.duplicated().sum())

# Drop duplicates
print(df.drop_duplicates(inplace=True))
print(df.duplicated().sum())
print(df.shape)

#Analyze Sentiment distribution
sentiment_counts = df['Sentiment'].value_counts()
print(sentiment_counts)

# Plot sentiment distribution
plt.figure(figsize=(6, 3))
sentiment_counts.plot(kind='bar', color=['red', 'green', 'yellow', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=0)
plt.show()

# Filter data for a specific brand (e.g., Microsoft)
brand_data = df[df['Entity'].str.contains('Microsoft', case=False)]
brand_sentiment_counts = brand_data['Sentiment'].value_counts()
print(brand_sentiment_counts)

# Plot sentiment distribution for the specific brand
plt.figure(figsize=(6, 6))
plt.pie(brand_sentiment_counts, labels=brand_sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution for Microsoft')
plt.show()