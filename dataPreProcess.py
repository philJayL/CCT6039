import csv

newRows = []
##grade conversions for KS2 grades
gradeConversion = {
    '2': 0,
    '2C': 0.25,
    '2B': 0.5,
    '2A': 0.75,
    '3': 1,
    '3C': 1.25,
    '3B': 1.5,
    '3A': 1.75,
    '4': 2,
    '4C': 2.25,
    '4B': 2.5,
    '4A': 2.75,
    '5': 3,
    '5C': 3.25,
    '5B': 3.5,
    '5A': 3.75,
    '6': 4,
    '6C': 4.25,
    '6B': 4.5,
    '6A': 4.75
}

with open(str('AllResults.csv')) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    gOffsets = {'A': 0.75, 'B': 0.5, 'C': 0.25}
    for row in csv_reader:
        missingData = False
        newRow = row
        ##skip headers
        if line_count != 0:
            ## set hml to numeric 1=L,2=M,3=H
            if row[4] != '':
                if row[4] == 'L':
                    newRow[4] = 1
                if row[4] == 'M':
                    newRow[4] = 2
                if row[4] == 'H':
                    newRow[4] = 3
            else:
                missingData = True
            ##convert reading 10 to numeric
            if row[5] in gradeConversion:
                newRow[5] = gradeConversion[row[5]]
            else:
                missingData = True
            ##convert Writing 10 to numeric
            if row[6] in gradeConversion:
                newRow[6] = gradeConversion[row[6]]
            else:
                missingData = True
            ##convert maths 10 to numeric
            if row[7] in gradeConversion:
                newRow[7] = gradeConversion[row[7]]
            else:
                missingData = True
            ## set engact to catagorical 1-9
            if row[12] != '' and row[12] != '0':
                newRow[12] = round(float(row[12]))
            else:
                missingData = True
            ## set mathact to catagorical 1-9
            if row[15] != '' and row[15] != '0':
                newRow[15] = round(float(row[15]))
            else:
                missingData = True
        if missingData == False:
            newRows.append(newRow)
        line_count += 1

##write new rows to a new file
with open('AllResultsPreProcessed.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(newRows)

