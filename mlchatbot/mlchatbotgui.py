import nltk
from tkinter import *
from tkinter import scrolledtext
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
stemmer = LancasterStemmer()
import os
import numpy
import tflearn
import tensorflow
import random
import time
import pickle
import json
window = Tk()
window.title("Chatbot App")
lbl1 = scrolledtext.ScrolledText(window,width=100,height=40,wrap=WORD)
lbl1.grid(column=0, row=1)
window.geometry('900x900')
window.configure(bg='#003152')
txt1 = Entry(window,width=60)
txt1.grid(column=0, row=2)

def hitenter(self):
    clicked()

def clicked():
    #responses = []
    inp = txt1.get()
    if inp.lower() == "quit":
        window.destroy()
    #print(words)
    results = model.predict([bag_of_words(inp, words)])
    print(inp)
    print('results = ')
    print(results)
    results_index = numpy.argmax([results])
    print('n.argmax =')
    print(results_index)
    tag = labels[results_index]
    print(tag)
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    lbl1.insert(INSERT,"\n\n")
    lbl1.insert(INSERT, "YOU: " + inp + "\n\n\t") 
    lbl1.insert(INSERT, "CHATBOT: ")           
    lbl1.insert(INSERT,responses)
    lbl1.insert(INSERT,"\n\n")
    txt1.delete(0, END)
    lbl1.see("end")

btn = Button(window, text="Chat",command=clicked)
btn.configure(bg='white')
btn.grid(column=1, row=2)
window.bind('<Return>', hitenter)

btnexit = Button(window, text="Exit",command=lambda: window.destroy())
btnexit.configure(bg='white')
btnexit.grid(column=2, row=2)


""" with open(os.path.abspath("C:/Users/linds/rfinalchatbot/intents.json"),encoding="utf8") as file:
    data = json.load(file) """
#nltk.download('stopwords')
#print(data)            prints the whole file
#print(data["intents"]) prints the dictionary
""" try:
    with open("./data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    
except: """
words = []
labels = []
docs_x = []
docs_y = []
""" 
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"]) """
with open(os.path.abspath("C:/Users/linds/rfinalchatbot/test.json"),encoding="utf8") as file:
    data = json.load(file)
for intent in data["intents"]:
    #for pattern in intent["patterns"]:
    wrds = nltk.word_tokenize(intent["patterns"])
    words.extend(wrds)
    docs_x.append(wrds)
    docs_y.append(intent["tag"])
    #if intent["tag"] not in labels:
    labels.append(intent["tag"])
words = [stemmer.stem(w.lower()) for w in words if w not in set(stopwords.words("english"))]
words = sorted(list(set(words)))
print(words)
labels = sorted(labels)
#print(labels)  #prints a list of our tags from intents file
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)  
print(training)
training = numpy.array(training)
output = numpy.array(output)
print(labels)
print(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 20)
#net = tflearn.dropout(net, 0.1)
net = tflearn.fully_connected(net, 10)
#net = tflearn.dropout(net, 0.01)
net = tflearn.fully_connected(net, 20)
#net = tflearn.dropout(net, 0.1)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net) 


""" try:
    model.load(os.path.abspath("C:/Users/linds/rfinalchatbot/model.tflearn")) 
except: """
model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
model.save(os.path.abspath("C:/Users/linds/rfinalchatbot/model.tflearn"))


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        print(se)
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    print(numpy.array(bag))        
    return numpy.array(bag)


lbl1.insert(INSERT,"Start talking with the bot (type quit to stop)!\n")
txt1.focus()
window.mainloop()
