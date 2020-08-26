import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

time = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
speed = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

speed_calc = np.poly1d(np.polyfit(time, speed, 5))

lin_space = np.linspace(1, 22, 100)

relationship = r2_score(speed, speed_calc(time))
print(relationship)

speed_prediction = speed_calc(5)
print(speed_prediction)

plt.scatter(time, speed)
plt.plot(lin_space, speed_calc(lin_space))
plt.show()
