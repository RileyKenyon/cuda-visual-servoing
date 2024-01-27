import matplotlib.pyplot as plt
from numpy import loadtxt

dataFile = open("output.txt","r")
data = dataFile.readlines()
data0 = data[0].split(',')
data1 = data[1].split(',')
data2 = data[2].split(',')
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(data0)
plt.subplot(3,1,2)
plt.plot(data1)
plt.subplot(3,1,3)
plt.plot(data2)
plt.show()
dataFile.close()
