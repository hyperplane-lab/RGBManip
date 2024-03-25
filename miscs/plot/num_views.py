import matplotlib.pyplot as plt
import numpy as np

# Number of views
x = [2, 3, 4, 5]
# OpenCabinetTrain
y1 = [84.25, 88.125, 86.125, 89.125]
# OpenCabinetTest
y2 = [78.5, 83.1875, 89.1875, 89.625]
# OpenDrawerTrain
y3 = [82.375, 81.0, 83.5, 83.0]
# OpenDrawerTest
y4 = [83.25, 85.375, 87.5, 87.0]

x = np.array(x)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y4 = np.array(y4)

# plt.plot(x, y1, label="OpenCabinetTrain")
# plt.plot(x, y2, label="OpenCabinetTest")

# plt.plot(x, y3, label="OpenDrawerTrain")
# plt.plot(x, y4, label="OpenDrawerTest")
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width - width) / 2
plt.bar(x, y1, width=width, label="DoorTrain")
plt.bar(x + width, y2, width=width, label="DoorTest")
plt.bar(x + 2*width, y3, width=width, label="DrawerTrain")
plt.bar(x + 3*width, y4, width=width, label="DrawerTest")

# Set x axis to integer only
plt.xticks(np.arange(2, 6, 1.0))
plt.yticks(np.arange(0, 101, 5.0))
plt.ylim(75, 90)

plt.xlabel("Number of Views")
plt.ylabel("Success Rate (%)")
plt.title("Number of Views and Success Rate")
plt.legend()
# plt.show()
plt.savefig("num_views.pdf")