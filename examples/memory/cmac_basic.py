import numpy as np
import matplotlib.pyplot as plt

from neupy import algorithms, environment


environment.reproducible()
plt.style.use('ggplot')

X_train = np.reshape(np.linspace(0, 2 * np.pi, 100), (100, 1))
X_test = np.reshape(np.sort(2 * np.pi * np.random.random(50)), (50, 1))

y_train = np.sin(X_train)
y_test = np.sin(X_test)

cmac = algorithms.CMAC(
    quantization=100,
    associative_unit_size=10,
    step=0.2,
    verbose=True,
    show_epoch=100,
)
cmac.train(X_train, y_train, epochs=100)
predicted_test = cmac.predict(X_test)

plt.plot(X_train, y_train)
plt.plot(X_test, predicted_test)
plt.show()
