import json

with open("./simple.json") as f:
  data = json.load(f)

for intent in data["intents"]:
    #print(intent)
    print(intent["patterns"])
    #for pattern in intent["patterns"]:
        #print(pattern)9+