import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

st.title('SMS Spam Classification')

# Load the dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    df = pd.read_csv(url, sep='\t', header=None)
    df.columns = ['labels', 'data']
    df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

# Data Preprocessing
Z = df['b_labels'].values
GP_vectorize_count = CountVectorizer(decode_error='ignore')
K = GP_vectorize_count.fit_transform(df['data'])
Ktrain, Ktest, Ztrain, Ztest = train_test_split(K, Z, test_size=0.33, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(Ktrain, Ztrain)

# # Display Training and Test Scores
# st.write("Train Score:", model.score(Ktrain, Ztrain))
# st.write("Test Score:", model.score(Ktest, Ztest))

def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=1600, height=800).generate(words)
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud)
    plt.axis('off')
    st.pyplot(plt)

# # Visualization
# st.write("Spam Word Cloud")
# visualize('spam')
# st.write("Ham Word Cloud")
# visualize('ham')

# User input for spam classification
st.write("### Check your own message")
user_message = st.text_input("Enter a message to check if it is spam or not:")

if user_message:
    user_message_vectorized = GP_vectorize_count.transform([user_message])
    prediction = model.predict(user_message_vectorized)[0]
    if prediction == 1:
        st.write("The message is classified as **Spam**.")
    else:
        st.write("The message is classified as **Ham**.")
