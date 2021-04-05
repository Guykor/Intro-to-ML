import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_excel("runtimes.xlsx")
data = data.set_index(data.columns[0])
data = pd.DataFrame(np.exp(np.divide(data, 50)))
data.plot()
plt.title("Elapsed run time for each model to train and \n"
          "predict as a function of sample size")
plt.xlabel("Sample size")
plt.ylabel("Run time (seconds), exp scaled")
plt.show()
