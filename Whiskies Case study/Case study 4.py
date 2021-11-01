# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 08:18:17 2021

@author: Ali
"""
import pandas as pd

import numpy as np


whisky = pd.read_csv("whiskies.txt")

whisky["Region"] = pd.read_csv("regions.txt")

flavors = whisky.iloc[:, 2:14]

corr_flavors = pd.DataFrame.corr(flavors)

import matplotlib.pyplot as plt

plt.figure(figsize= (10,10))

plt.pcolor(corr_flavors)

plt.colorbar()



corr_whisky = pd.DataFrame.corr(flavors.transpose())

plt.figure(figsize= (10,10))

plt.pcolor(corr_whisky)

plt.colorbar()
plt.axis("tight")

from sklearn.cluster import SpectralCoclustering

model = SpectralCoclustering(n_clusters =6, random_state = 0)

model.fit(corr_whisky)

"""np.sum(model.rows_, axis = 1)

model.row_labels_"""


whisky["Group"] = pd.Series(model.row_labels_, index= whisky.index)

whisky = whisky.iloc[np.argsort(model.row_labels_)]

whisky = whisky.reset_index(drop=True)

correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())
correlations = np.array(correlations)


plt.figure(figsize= (14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title("Original")

plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")

plt.savefig("correlations.pdf")
