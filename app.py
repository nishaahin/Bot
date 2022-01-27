import streamlit as st
from streamlit_chat import message
import requests
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer 
import json


import string # to process standard python strings
import warnings
import re
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pickle

nltk.download('popular', quiet=True)

# Reading Dataframe
df = pd.read_csv('Health Care Chatbot -FAQ.csv', encoding='ISO-8859-1')

data = ''.join(df.Questions).split('.')

# Cleaning Data
def clean_data(text):
    '''
    Function to clean the text and remove url, usernames and special characters
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())


df['clean_text'] = df.Questions.apply(lambda x: clean_data(x))
clean_text = df.clean_text.values

# Lemmatizing Text
def lemmatizingText(text):
    '''
    Function to lemmatize the string
    '''
    
    # Splitting into Tokens
    text = text.split()  # Tokenization

    # Removing Stopwords and Lemmatization
    
    # Initilzing lametizer
    lemma = WordNetLemmatizer()
    # Stopwords set
    stopword = set(stopwords.words('english'))
    
    # removing stopwords from sentenses, lower casing the words and lemmatizing them
    text = [lemma.lemmatize(word.lower(), 'v') for word in text if not word in stopword]
    return ' '.join(text)

# calling sentenses one by one and storing in a list again
df['lemma_text'] = df.clean_text.apply(lambda x: lemmatizingText(x))


# Vectorizing
tfidf = TfidfVectorizer(max_features=300)

corpus = df.lemma_text.values

# Training
vec = tfidf.fit_transform(corpus)




def get_text():
    input_text = st.text_input("You: ","So, what's in your mind")
    return input_text


def get_response(user_inp):
       
    # Cleaning & Processing user input
    user_inp = lemmatizingText(clean_data(user_inp))
    inp_vec = tfidf.transform([user_inp])
    
    # Finding queries similar to user's query based on cosine similarity
    vals = cosine_similarity(inp_vec, vec)
#     return vals

    # Greetings
    if user_inp.lower() in ['hi','hello','hey']:
        return random.choice(['hi','hello','hey','Hey There!'])
        
    # How are you
    elif user_inp.lower() in ['how you doing?','how are you doing?','how you doing','how are you doing?','how are you','how are you?']:
        return 'Doing good, Thanks for asking! <br> Hope you are doing well, How can i help you?'
        
    elif vals.sum() == 0:
        return 'Sorry, bot is not able to understand that currently.'
    else:
        idx = vals.argsort()[0][-1]
        return df.Answers[idx]
    

def find_senitment(text):
    from textblob import TextBlob
    text = TextBlob(text)
    polarity = text.polarity

    if polarity == 0:
        return 'Neutral'

    elif polarity > 0:
        return 'Positive'

    else:
        return 'Negative'
    
col1, col2 = st.columns([1,3])
with col1:
    st.image('logo.jpg')

with col2:
    st.sidebar.title("NLP Bot")
    st.title("""
    Healthcare Bot based on Natural Language Processing
    Healthcare Bot is an NLP conversational chatterbot. Initialize the bot by clicking the "Initialize bot" button. 
    """)


ind = 1
if st.sidebar.button('Initialize bot'):
    #do something
    
    st.title("Your bot is ready to talk to you")
    ind = ind +1
        
user_input = get_text()

conversation = {}

if True:
    resp = get_response(user_input)
    st.text_area("Bot:", value=resp, height=200, max_chars=None, key=None)
    conversation[user_input] = resp
else:
    st.text_area("Bot:", value="Please start the bot by clicking sidebar button", height=200, max_chars=None, key=None)

st.sidebar.header('Sentiment : '+find_senitment(user_input))
st.sidebar.write(conversation)
