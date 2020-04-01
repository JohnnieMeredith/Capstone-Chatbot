import csv
import json

csvfile = open('QA.tsv', 'r')
jsonfile = open('QAjson.json', 'w')

fieldnames = ("Intent","Tag", "Pattern")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')