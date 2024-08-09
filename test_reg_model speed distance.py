import random
import pandas as pan
import numpy as np
#creating a data for speed or dist

sp=[random.randint(10,1000) for i in range(1000) ]
ds=[random.randint(100,10000) for i in range(1000) ]
sp.sort()
ds.sort()

data=pan.DataFrame({'speed':sp,'dist':ds})
print(data)

data.to_csv('speed_dist.csv',index=False)

#visualization
import matplotlib.pyplot as plt

#open or use that data / clean using drop when data are nan
data=pan.read_csv('speed_dist.csv')
data.dropna(inplace=True)
data.reset_index(inplace=True) #old or new index
data.drop(['index'],axis=1,inplace=True)
print(data)
	
x=data['speed']
y=data['dist']
plt.plot(data['speed'],data['dist'])
plt.show()

#data division on train data and test data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.97)
print(x_train,x_test)

#create model
from sklearn import linear_model

#load linear regression
model=linear_model.LinearRegression()

#data making as per model requirement
x_train=np.array(x_train).reshape(-1,1)
x_test=np.array(x_test).reshape(-1,1)


#train linear on data
model.fit(x_train,y_train)

#predict data on test

print("predict data:",model.predict(x_test))

print("actual data",list(y_test))

print('model accuracy',model.score(x_test,y_test))









