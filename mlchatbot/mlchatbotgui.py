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
import time


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
    lbl1.insert(END,"\n\n")
    lbl1.insert(END, "YOU: \n" + inp + "\n\n\n", 'tag-left')
    lbl1.insert(END, "CHATBOT: \n",'tag-right')           
    lbl1.insert(END,random.choice(responses),'tag-right')
    lbl1.insert(END,"\n\n\n")
    txt1.delete(0, END)
    lbl1.see("end")
    
def chatbot():
    global words
    global data
    global training
    global output
    global labels
    global model
    
    #nltk.download('stopwords')
    #print(data)            prints the whole file
    #print(data["intents"]) prints the dictionary
    words = []
    labels = []
    docs_x = []
    docs_y = []
    training = []
    output = []
    
   
    with open("QAjson.json",encoding="utf8") as file:
        data = json.load(file)
    for intent in data["intents"]:
        #for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(intent["patterns"])
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        #if intent["tag"] not in labels:
        labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w not in set(stopwords.words("english")) and w != '?']
    words = sorted(list(set(words)))
    print(words)
    #print(len(words))
    #time.sleep(5000)
    labels = sorted(labels)
    #print(labels)  #prints a list of our tags from intents file
    

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc if w not in set(stopwords.words("english")) and w != '?']

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)  
            
    training = numpy.array(training)
    output = numpy.array(output)
        
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
    NAME = "Capstone-Quizbot{}".format(int(time.time()))
    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])], name='InputData')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 10, name='Layer_1')
    net = tflearn.fully_connected(net, 10, name='Layer_2')
    #net = tflearn.dropout(net, 0.2)
    net = tflearn.fully_connected(net, 10, name='Layer_3')
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net, loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_dir='/tfboardlog/{}'.format(NAME),tensorboard_verbose = 3) 


    try:
        model.load("./model.tflearn")
    except:
        model.fit(training, output, n_epoch=500, batch_size=8,validation_set=0.1, show_metric=True, shuffle=True)
        model.save("./model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words if word not in set(stopwords.words("english")) and word != '?']

    for se in s_words:
        print(se)
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    print(numpy.array(bag))        
    return numpy.array(bag)


chatbot()

window = Tk()
window.title("Chatbot App")
window.configure(bg= '#B8BCFF')
window.geometry('1300x900')
window.bind('<Return>', hitenter)
helv36 = tkFont.Font(family='Helvetica', size=20, weight='bold')
lbl1 = scrolledtext.ScrolledText(window,width=80,height=20,wrap=WORD)
lbl1["font"] = helv36
lbl1.grid(column=0,padx=20,pady=10,row=1,columnspan = 3,sticky=NW)

txt1 = Entry(window, width=50)
txt1["font"] = helv36
txt1.grid(column=0,padx=20,pady=10, row=2)
btn = Button(window, text="Chat",padx = 45, pady = 20, command=clicked)
btn['font'] = helv36
btn.configure(bg='#D1D4FF')
btn.grid(column=1, ipadx=0,ipady=0, row=2)

btnexit = Button(window, text="Exit",padx = 45, pady = 20, command=lambda: window.destroy())
btnexit['font'] = helv36
btnexit.configure(bg='#D1D4FF')
btnexit.grid(column=2, row=2)
lbl1.insert(INSERT,"Start talking with the bot (type quit to stop)!\n")
txt1.focus()
window.mainloop()
