import csv
import json

csvfile = open('file.csv', 'r')
jsonfile = open('file.json', 'w')

fieldnames = ("QuestionID","Question","DocumentID","DocumentTitle","SentenceID","Sentence","Label")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')