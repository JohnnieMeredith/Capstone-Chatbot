import csv
import json

csvfile = open('file.tsv', 'r')
jsonfile = open('onlyfiledumps.json', 'w')

fieldnames = ("QuestionID","Question","DocumentID","DocumentTitle","SentenceID","Sentence","Label")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(jsonfile,row)
    jsonfile.write('\n')