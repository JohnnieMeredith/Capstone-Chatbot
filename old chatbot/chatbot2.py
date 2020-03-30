from newspaper import Article
import random
import string
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
import gzip
import nltk
import json
import numpy as np
import warnings
import re, string, unicodedata

warnings.filterwarnings('ignore')
#nltk.download()
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

article = Article('https://en.wikipedia.org/wiki/Neoliberalism')
article.download()
article.parse()
article.nlp()
corpus = article.text

#Print the corpus/text
#print(corpus)

#Tokenization
text = corpus
sent_tokens = nltk.sent_tokenize(text) #Convert the text into a list of sentences

#Print the list of sentences
#print(sent_tokens)

#Create a dictionary (key:value) pair to remove punctuations
remove_punct_dict = dict(  ( ord(punct),None) for punct in string.punctuation)

#Print the punctuations
#print(string.punctuation)

#Print the dictionary
#print(remove_punct_dict)

#Create a function to return a list of lemmatized lower case words after removing punctuations
def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    #word tokenization
    word_token = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
    
    #remove ascii
    new_words = []
    for word in word_token:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    
    #Remove tags
    rmv = []
    for w in new_words:
        text=re.sub("&lt;/?.*?&gt;","&lt;&gt;",w)
        rmv.append(text)
        
    #pos tagging and lemmatization
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    json.dumps({'id' : tag_map})
    lmtzr = WordNetLemmatizer()
    lemma_list = []
    rmv = [i for i in rmv if i]
    for token, tag in nltk.pos_tag(rmv):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemma_list.append(lemma)
    return lemma_list

#Print the tokenization text
#print(LemNormalize(text))


#Greeting Inputs
GREETING_INPUTS = ["hi", "hello", "hola", "greetings", "wassup", "hey"]

#Greeting responses back to the user
GREETING_RESPONSES=["howdy", "hi", "hey", "what's good", "hello", "hey there"]

#Function to return a random greeting response to a users greeting
def greeting(sentence):
  #if the user's input is a greeting, then return a randomly chosen greeting response
  for word in sentence.split():
    if word.lower() in GREETING_INPUTS:
      return random.choice(GREETING_RESPONSES)

      #Generate the response
def response(user_response):
  

  #The users response / query
  #user_response = 'What is chronic kidney disease'

  user_response = user_response.lower() #Make the response lower case

  ###Print the users query/ response
  #print(user_response)

  #Set the chatbot response to an empty string
  robo_response = ''

  #Append the users response to the sentence list
  sent_tokens.append(user_response)

  ###Print the sentence list after appending the users response
  #print(sent_tokens)

  #Create a TfidfVectorizer Object
  TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words='english')

  #Convert the text to a matrix of TF-IDF features
  tfidf = TfidfVec.fit_transform(sent_tokens)

  ###Print the TFIDF features
  #print(tfidf)

  #Get the measure of similarity (similarity scores)
  vals = cosine_similarity(tfidf[-1], tfidf)

  #Print the similarity scores
  #print(vals)

  #Get the index of the most similar text/sentence to the users response
  idx = vals.argsort()[0][-2]
  #print("this is idx")
  #print(idx)
  #Reduce the dimensionality of vals
  flat = vals.flatten()

  #sort the list in ascending order
  flat.sort()
  #print("This is flat.sort")
  """ for x in flat:
    print(x) """
  #Get the most similar score to the users response
  score = flat[-2]

  #Print the similarity score
  print(score)

  #If the variable 'score' is 0 then their is no text similar to the users response
  if(score == 0):
    robo_response = robo_response+"I apologize, I don't understand."
  else:
    robo_response = robo_response+sent_tokens[idx]
  
  #Print the chat bot response
  #print(robo_response)
  
  #Remove the users response from the sentence tokens list
  sent_tokens.remove(user_response)
  
  return robo_response


flag = True
print("MyBot: I am a chatbot. I will answer your queries about things. If you want to exit, type Bye!")
while(flag == True):
  user_response = input()
  user_response = user_response.lower()
  if(user_response != 'bye'):
    if(user_response == 'thanks' or user_response =='thank you'):
      flag=False
      print("MyBot: You are welcome !")
    else:
      if(greeting(user_response) != None):
        print("MyBot: "+greeting(user_response))
      else:
        print("MyBot: "+response(user_response))       
  else:
    flag = False
    print("MyBot: Chat with you later !")