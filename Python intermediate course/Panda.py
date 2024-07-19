
import pandas as pd
starting = { 'col_1': [2,3,4,5],
             'col_2': [4,5,7,4],
             'col_3': [1,3,7,5],
             'col_4': [5,3,4,9],
             'col_5': [6,8,4,5],
             'col_6': [7,9,4,9]}
df=pd.DataFrame(starting)
print(df)
print(df.dtypes)
print(df.col_1)
print(df['col_1'])

# us ehead , slice and tail
import pandas as pd
starting1 = { 'col_1': [2,5,4,5,5, 6],
             'col_2': [4,5,5,4,4, 5],
             'col_3': [1,3,7,5,6,7],
             'col_4': [5,3,4,9,3,6],
             'col_5': [6,8,4,5,5,6],
             'col_6': [7,9,4,9,3,8]}
df=pd.DataFrame(starting1)
print(df.head(1))
print(df)
print(df.tail(1))
#Slice
import pandas as pd
starting2 = { 'col_1': [2,5,4,5,5, 6],
             'col_2': [4,5,9,4,4, 5],
             'col_3': [1,3,7,5,6,7],
             'col_4': [5,3,4,9,3,6],
             'col_5': [6,8,4,5,5,6],
             'col_6': [7,9,4,9,3,8]}
df=pd.DataFrame(starting2)
print(df.col_2[2])       # index value
print(df['col_2'][2])     #index value

# indexing
import pandas as pd
starting2 = { 'col_1': [2,5,4,5,5, 6],
             'col_2': [4,5,9,4,4, 5],
             'name': ['ali','hasan','saa','jdkf','sds','sdd'],
             'col_4': [5,3,4,9,3,6],
             'col_5': [6,8,4,5,5,6],
             'col_6': [7,9,4,9,3,8]}
df=pd.DataFrame(starting2)
df= df.set_index('name')
print(df.index)
# or you can write as
print(df.set_index('name'))
print(df)

# Read CSV data

import pandas as pd

df = pd.read_csv('test_csv.csv')
print(df.head())
print(len(df))
df=pd.DataFrame(df)
df=df.set_index('zip')
print(df.head())
print(len(df))
#
print(pd.DataFrame.count(df))

### read and count all csv data
row_count, column_count=df.shape
print(df.shape)
print(row_count)
print(column_count)

## IO operation

import pandas as pd
df=pd.read_csv('test_csv.csv', index_col=0)   #here index tell the 1st col became as a level
print(df.head())
df.to_csv('new_csv_file')    ### write a new csv file
ef=pd.read_csv('new_csv_file')
print(ef)


### use how to read json and write json
import pandas as pd
df=pd.read_csv('new_csv_file')
df.to_json('new_jason')    # write json file
es=pd.read_json('new_jason')    # read json file
print(es)

### colume multification
import pandas as pd

df = pd.read_csv('test_csv.csv')
df['zip']= df['zip']/10
df['zip']= df['zip']+10
print(df)

### compair two cvs colum of data

###  concatenation DataFrame

import pandas as pd
df1= pd.DataFrame({'temp ': [75,87,45],
                   'Huminity ': [5,7,5],
                   'per ': [7,8,4]},
                  index= [0,1,2])
df2= pd.DataFrame({'temp ': [705,807,405],
                   'Huminity ': [15,17,15],
                   'per ': [71,81,41]},
                  index= [3,4,5])
df3= pd.DataFrame({'temp ': [725,827,425],
                   'Huminity ': [53,73,53],
                   'per ': [74,84,44]},
                  index= [6,7,8])
conca=pd.concat([df1,df2,df3])
print(conca)
ap=df1.append(df2)
print(ap)


