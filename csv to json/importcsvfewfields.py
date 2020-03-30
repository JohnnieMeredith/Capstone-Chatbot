import csv
import json

csvfile = open('test.tsv', 'r', encoding = "utf8")
jsonfile = open('testjsonlessfields.json', 'w')

fieldnames = ("Label","Question","Sentence")
reader = csv.DictReader( csvfile,  delimiter = '\t' )
for row in reader:
    json.dumps(row)
    jsonfile.write('\n')