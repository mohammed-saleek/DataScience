import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

age = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

age_mean = np.mean(age)
print(age_mean)

age_median = np.median(age)
print(age_median)

age_mode = stats.mode(age)
print(age_mode)

standard_division = np.std(age)
print(standard_division)

variance = np.var(age)
print(variance)

speed_mean = np.mean(speed)
print(speed_mean)

speed_median = np.median(speed)
print(speed_median)

speed_mode = stats.mode(speed)
print(speed_mode)

speed_standard_division = np.std(speed)
print(speed_standard_division)

speed_variance = np.var(speed)
print(speed_variance)

slope, intercept, r, p, std_err = stats.linregress(age, speed)
print(slope)
print(intercept)
print(r)
print(p)
print(std_err)


def speed_calc(age):
    return slope * age + intercept


speed_value = list(map(speed_calc, age))

plt.scatter(age, speed)
plt.plot(age, speed_value)
prediction_speed = speed_calc(age=1)
print(prediction_speed)
plt.show()
