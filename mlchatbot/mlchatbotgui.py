import nltk
from tkinter import *
from tkinter import scrolledtext
from tkinter import font as tkFont
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
window.configure(bg= '#B8BCFF')
helv36 = tkFont.Font(family='Helvetica', size=20, weight='bold')
lbl1 = scrolledtext.ScrolledText(window,width=80,height=20,wrap=WORD)
lbl1["font"] = helv36
lbl1.grid(column=0,padx=20,pady=10,row=1,columnspan = 3,sticky=NW)
window.geometry('1300x900')
txt1 = Entry(window, width=50)
txt1["font"] = helv36
txt1.grid(column=0,padx=20,pady=10, row=2)

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
    lbl1.tag_configure('tag-left', justify='left')
    lbl1.tag_configure('tag-right', justify='right')
    #lbl1.tag_configure('color-black',fg='#ffffff')
    #lbl1.tag_configure('color-red', foreground='#ff0000')
    tag = labels[results_index]
    print(tag)
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    lbl1.insert(INSERT,"\n\n")
    lbl1.insert(INSERT, "YOU: \n" + inp + "\n\n\n", 'tag-left')
    lbl1.insert(INSERT, "CHATBOT: \n",'tag-right')           
    lbl1.insert(INSERT,random.choice(responses),'tag-right')
    lbl1.insert(INSERT,"\n\n\n")
    txt1.delete(0, END)
    lbl1.see("end")

btn = Button(window, text="Chat",padx = 45, pady = 20, command=clicked)
btn['font'] = helv36
btn.configure(bg='#D1D4FF')
btn.grid(column=1, ipadx=0,ipady=0, row=2)
window.bind('<Return>', hitenter)

btnexit = Button(window, text="Exit",padx = 45, pady = 20, command=lambda: window.destroy())
btnexit['font'] = helv36
btnexit.configure(bg='#D1D4FF')
btnexit.grid(column=2, row=2)


""" with open(os.path.abspath("C:/Users/Johnnie5Dubv/Documents/chatbot/Capstone-Chatbot/mlchatbot/json stuff/intents.json"),encoding="utf8") as file:
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
with open(os.path.abspath("C:/Users/Johnnie5Dubv/Documents/chatbot/Capstone-Chatbot/json stuff/QAjson.json"),encoding="utf8") as file:
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
    model.load(os.path.abspath("C:/Users/Johnnie5Dubv/Documents/chatbot/Capstone-Chatbot/mlchatbot/model.tflearn")) 
except: """
model.fit(training, output, n_epoch=1, batch_size=16, show_metric=True)
model.save(os.path.abspath("C:/Users/Johnnie5Dubv/Documents/chatbot/Capstone-Chatbot/mlchatbot/model.tflearn"))


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
