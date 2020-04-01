import csv
import json

csvfile = open('QA.tsv', 'r')
jsonfile = open('QAjson.json', 'w')

fieldnames = ("Pattern","Tag", "Response")
reader = csv.DictReader( csvfile, fieldnames, delimiter="\t")
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')