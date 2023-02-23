import csv
import os


def write_deleted_questions(rows, filename):
    print("writing deleted questions to " + filename)
    fields = ['question', 'answer', 'last gen question', 'last gen answer']

    # check for file - if there remove it, then create it either way
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w', encoding="utf-8") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
