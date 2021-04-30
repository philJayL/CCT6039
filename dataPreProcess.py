import csv

newRows = []

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
            if row[5] != '' and row[5] != 'N' and row[5] != 'B':
                grade = row[5]
                if grade[0] == '2':
                    newRow[5] = 0 + gOffsets[grade[1]]
                if grade[0] == '3':
                    newRow[5] = 1 + gOffsets[grade[1]]
                if grade[0] == '4':
                    newRow[5] = 2 + gOffsets[grade[1]]
                if grade[0] == '5':
                    newRow[5] = 3 + gOffsets[grade[1]]
                if grade[0] == '6':
                    newRow[5] = 4 + gOffsets[grade[1]]
            else:
                missingData = True
            ##convert Writing 10 to numeric
            if row[6] != '' and row[6] != 'N' and row[6] != 'B':
                grade = row[6]
                if grade[0] == '2':
                    if len(grade) > 1:
                        newRow[6] = 0 + gOffsets[grade[1]]
                    else:
                        newRow[6] = 0
                if grade[0] == '3':
                    if len(grade) > 1:
                        newRow[6] = 1 + gOffsets[grade[1]]
                    else:
                        newRow[6] = 1
                if grade[0] == '4':
                    if len(grade) > 1:
                        newRow[6] = 2 + gOffsets[grade[1]]
                    else:
                        newRow[6] = 2
                if grade[0] == '5':
                    if len(grade) > 1:
                        newRow[6] = 3 + gOffsets[grade[1]]
                    else:
                        newRow[6] = 3
                if grade[0] == '6':
                    if len(grade) > 1:
                        newRow[6] = 4 + gOffsets[grade[1]]
                    else:
                        newRow[6] = 4
            else:
                missingData = True
            ##convert maths 10 to numeric
            if row[7] != '' and row[7] != 'N' and row[7] != 'B':
                grade = row[7]
                if grade[0] == '2':
                    if len(grade) > 1:
                        newRow[7] = 0 + gOffsets[grade[1]]
                    else:
                        newRow[7] = 0
                if grade[0] == '3':
                    if len(grade) > 1:
                        newRow[7] = 1 + gOffsets[grade[1]]
                    else:
                        newRow[7] = 1
                if grade[0] == '4':
                    if len(grade) > 1:
                        newRow[7] = 2 + gOffsets[grade[1]]
                    else:
                        newRow[7] = 2
                if grade[0] == '5':
                    if len(grade) > 1:
                        newRow[7] = 3 + gOffsets[grade[1]]
                    else:
                        newRow[7] = 3
                if grade[0] == '6':
                    if len(grade) > 1:
                        newRow[7] = 4 + gOffsets[grade[1]]
                    else:
                        newRow[7] = 4
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

