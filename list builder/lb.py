import json
import os
question = ''
answer = ''
tag = ''
questions = []
answers = []
tagline = []
words = []
labels = []
docs_x = []
docs_y = []

with open(os.path.abspath("C:/Users/Melanie/chatbot/ml chatbot/test.json"),encoding="utf8") as file:
    data = json.load(file)
for intent in data["intents"]:
    #for pattern in intent["patterns"]:
    if intent["tag"] in tagline:
        question.append(intent["patterns"])
        answer.append(intent["responses"])
    else:
        tag.append(intent["tag"])
        tagline.append(intent["tag"])
    