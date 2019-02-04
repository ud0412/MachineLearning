import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

plt.plot(train_x, train_y, 'o')
plt.show()

theta0 = np.random.rand()
theta1 = np.random.rand()

def f(x):
  return theta0 + theta1 * x

def E(x, y):
  return 0.5 * np.sum((y - f(x)) ** 2)

mu = train_x.mean()
sigma = train_x.std()

def standardize(x):
  return (x - mu) / sigma

train_z = standardize(train_x)

plt.plot(train_z, train_y, 'o')
plt.show()

ETA = 1e-3
diff = 1
count = 0

error = E(train_z, train_y)
while diff > 1e-2:
  tmp0 = theta0 - ETA * np.sum((f(train_z) - train_y))
  tmp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)

  theta0 = tmp0
  theta1 = tmp1

  current_error = E(train_z, train_y)
  diff = error - current_error
  error = current_error

  count += 1
  log = '{}회째: theta0 = {:.3f}, theta1 = {:.3f}, 차분={:.4f}'
  print(log.format(count, theta0, theta1, diff))

x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show()

