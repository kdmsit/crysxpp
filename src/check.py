import os
import csv
import pandas as pd


id_prop_file = os.path.join('../data/id_prop.csv')
id_prop_data=[]
with open(id_prop_file) as f:
    reader = csv.reader(f)
    for row in reader:
        id_prop_data.append([row[0],row[1]])
with open('id_prop.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(id_prop_data)

# df = pd.DataFrame(id_prop_data)
# df.to_csv('id_prop.csv')