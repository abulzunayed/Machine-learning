
import matplotlib.pyplot as plt
x=[1,2,3,10,11]
y=[5,6,2,79,80]
#plt.plot([1,2,3,10,11], [5,6,2,79,80])
plt.plot(x, y)### always x and y dimension must be same .
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title('this is python plot title. \n show as pcture')  # '\n' indicate the new line
plt.show()
print(len(x))
print(len(y))
## use legends function. It make subplot and make corner top box which defind ech of line.
import matplotlib.pyplot as plt
x=[1,2,3,10,11]
y=[5,6,2,79,80]
y2=[2,4,6,7,9]
plt.plot(x, y, label= 'Intitial line')
plt.plot(x, y2, label= 'New line')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title('this is python plot title. \n show as pcture')  # '\n' indicate the new line
plt.legend()   #### define each line name by top box.
plt.show()


### use Bar chart.
import matplotlib.pyplot as plt
x,x2, x3=[1,2,3,10,11],[5,7,9,11,13],[5,7,9,11,13]
y=[5,6,2,79,80]
y2=[2,4,6,7,9]
plt.bar(x,y, label='one', color= 'r')
plt.bar(x2,y2, label='Two', color= 'g')
plt.bar(x3,y, label='Two', color= 'b')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()

### Use Histrogram
import matplotlib.pyplot as plt

x=[22,55,66,77,88,33, 44, 66,77,99,11]
y=[10,20,30,40,50,60,70,80,90,100]
plt.hist(x,y,histtype='bar', cumulative=False,rwidth=.5)
plt.show()

## scatter plot.
import matplotlib.pyplot as plt
x=[10,50,70,10,50,50,90,30,90,100]
y=[10,20,30,40,50,60,70,80,90,100]
plt.scatter(x,y, c='r', marker='o')
plt.show()

### Use Pie chart

import matplotlib.pyplot as plt
labels='taxes', 'overhead', 'entertainment'
x=[25,32,25]
colors= ['r','g', 'b']
plt.pie(x,labels=labels, colors=colors)
plt.show()

##loading data from CSV
import matplotlib.pyplot as plt
import csv
x = []
y = []
with open('CSV_file_create.txt', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(int(row[1]))

plt.plot(x, y, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()


