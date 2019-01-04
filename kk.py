import nltk
import numpy as np
import random
import string # to process standard python strings
import csv
import pandas as pd 
import numpy as np
filename = 'CAvideos'
f=open('corpus.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase
nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words
sent_tokens[:2]
['a chatbot (also known as a talkbot, chatterbot, bot, im bot, interactive agent, or artificial conversational entity) is a computer program or an artificial intelligence which conducts a conversation via auditory or textual methods.',
 'such programs are often designed to convincingly simulate how a human would behave as a conversational partner, thereby passing the turing test.']
word_tokens[:2]
['a', 'chatbot', '(', 'also', 'known']
lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

df = pd.read_csv(r"CAvideos.csv",encoding = "ISO-8859-1")
def getMostLikedVideo(data):
# a = max(df["likes"])
# print(a)
# count = 0
# for row in df["likes"]:
#     count += 1
#     if row == a:
#         break
# print(count)
    print(df.loc[np.argmax(df[data])])

def getMostViewedVideo(data):
    print(df.loc[np.argmax(df[data])])

def getMostCommentedVideo(data):
    print(df.loc[np.argmax(df[data])])

def getMostDislikedVideo(data):
    print(df.loc[np.argmax(df[data])])

def getVideoWithId(data):
    pass


def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import brown
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                # print("ROBO: ",end="")
                # print(response(user_response))
                # sent_tokens.remove(user_response)
                # tokens = nltk.word_tokenize(sentence)
                sentence = nltk.word_tokenize(user_response)
                for word in sentence:
                    if(word == 'likes' or word == 'like'):
                        getMostLikedVideo('likes')
                    elif(word == 'dislikes' or word == "dislike"):
                        getMostDislikedVideo('dislikes')
                    elif(word == 'comments' or word =='comment'):
                        getMostCommentedVideo('comment_count')
                    elif(word == 'views'):
                        getMostViewedVideo("views")
                print(sentence)

    else:
        flag=False
        print("ROBO: Bye! take care..")