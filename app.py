import pandas as pd
import numpy as np
import streamlit as st
import joblib
from PIL import Image
import pickle
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import re
import string
import matplotlib.pyplot as plt
import time



@st.cache(allow_output_mutation=True)
def load(vectoriser_path, model_path):

    # Load the vectoriser.
    file = open(vectoriser_path, 'rb')
    vectoriser = pickle.load(file)
    file.close()
    
    # Load the LR Model.
    file = open(model_path, 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel






def inference(vectoriser, model, tweets, cols):

    text = tweets.split(";")
    
    finaldata = []

    textdata = vectoriser.transform(lemmatize_process(preprocess(text)))
    sentiment = model.predict(textdata)

    sentiment_prob = model.predict_proba(textdata)
    
    for index,tweet in enumerate(text):
        if sentiment[index] == 1:
            sentiment_probFinal = sentiment_prob[index][1]
        else:
            sentiment_probFinal = sentiment_prob[index][0]
            
        sentiment_probFinal2 = "{}%".format(round(sentiment_probFinal*100,2))
        finaldata.append((tweet, sentiment[index], sentiment_probFinal2))
           
    df = pd.DataFrame(finaldata, columns = ['Tweet','Sentiment', 'Probability(Confidence Level)'])
    df = df.replace([0,1], ["Negative","Positive"])
    
    return df

def get_wordnet_pos_tag(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

def lemmatize_process(preprocessedtext):
    # Create Lemmatizer
    lemma = WordNetLemmatizer()
    
    finalprocessedtext = []
    for tweet in preprocessedtext:
        text_pos = pos_tag(word_tokenize(tweet))
        words = [x[0] for x in text_pos]
        pos = [x[1] for x in text_pos]
        tweet_lemma = " ".join([lemma.lemmatize(a,get_wordnet_pos_tag(b)) for a,b in zip(words,pos)])
        finalprocessedtext.append(tweet_lemma)
    return finalprocessedtext        


def plot(df):
    positive = round(np.count_nonzero(df['Sentiment'] == "Positive")/len(df['Sentiment'])*100,2)
    negative = round(np.count_nonzero(df['Sentiment'] == "Negative")/len(df['Sentiment'])*100,2)
    
    labels = ['Positive','Negative']
    values = np.array([positive,negative])
    myexplode = [0.2, 0]
    mycolors = ["green", "red"]
    
    fig,ax = plt.subplots()
    ax.pie(values, labels = labels, explode = myexplode, shadow = True, colors = mycolors)
    ax.legend()
    ax.set_title("Positive vs Negative Tweet(%)")
    st.pyplot(fig)


def preprocess(textdata):

    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
              ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

    stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                 'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before', 'but' ,
                 'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
                 'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
                 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                 'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                 'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
                 'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
                 's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
                 't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
                 'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
                 'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
                 'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
                 "youve", 'your', 'yours', 'yourself', 'yourselves']


    processedText = []
    
    wordLemm = WordNetLemmatizer()
    
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
    
        tweet = re.sub(urlPattern,' URL',tweet)
        

        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])    
            
        tweet = re.sub(userPattern,' USER', tweet)  
        
        tweet = re.sub(alphaPattern, " ", tweet)
        
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        
        all_char_list = []
        all_char_list = [char for char in tweet if char not in string.punctuation]
        tweet = ''.join(all_char_list)
        
        tweetwords = ''
        for word in tweet.split():
            if word not in (stopwordlist):
                if len(word)>1:
                    tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText


def progressbar():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)





st.title('Sentiment Analysis App')
st.write('Performing Sentiment Analysis')
image = Image.open('data/sentiment.jpg')
st.image(image, use_column_width=True)

st.sidebar.subheader("Enter single/multiple tweets separated by semicolon : ")
tweets = st.sidebar.text_area("Some samples are provided below for reference..", value="I hate twitter;I do not like the movie;Mr. Stark, I don't feel so good;May the Force be with you.;I read the book, the content is not good;This is a new beginning for us", height=500, max_chars=None, key=None)
cols = ["tweet"]

    
if (st.sidebar.button('Predict Sentiment')):   
    progressbar()
    
    vectoriser, model = load('models/vectoriser.pickle', 'models/Sentiment-LR.pickle')
    result_df = inference(vectoriser, model, tweets, cols)
    st.table(result_df)
    st.text("")
    st.text("")
    st.text("")
    plot(result_df)
    

# data = pd.read_csv("data/real_data.csv", encoding = "ISO-8859-1")

