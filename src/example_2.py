import Data
import SupervisedModel
from Correction import corriger
import matplotlib.pyplot as plt
import numpy as np

data = Data.Data()
data.RECORD_SECONDS = 10

data.record()
plt.plot(data.amplitude)

peaks = Data.get_peaks_from_click(data.amplitude, data.clicks, 4000)
plt.scatter(peaks, np.array(data.amplitude)[peaks], c='r')
plt.show()


data.save_json()



