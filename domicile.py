# import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#input_file="resultats.csv"
input_file="resultats-epures2.csv"
df1 = pd.read_csv(input_file, header = 0)
df1 = df1-3

# Scale data befor applying PCA
scaling=StandardScaler()
  
# Use fit and transform method 
scaling.fit(df1)
Scaled_data=scaling.transform(df1)
  
# Set the n_components=3
principal=PCA(n_components=3)
principal.fit(Scaled_data)
#x=principal.transform(Scaled_data)
x =principal.transform(df1)

donnees = pd.read_csv("donnees.csv", header = 0)

# 2 composantes principales

# Sexe

fig, ax = plt.subplots()

CustomCmap = ListedColormap(['yellow','blue','red','black'])
scatter = ax.scatter(x[:,1],x[:,2],c=donnees['c_habitat'],cmap=CustomCmap)

sans = mpatches.Patch(color='yellow', label='Pas de réponse')
hameau = mpatches.Patch(color='blue', label='Hameau')
village = mpatches.Patch(color='red', label='Village')
ville = mpatches.Patch(color='black', label='Ville')

plt.legend(handles=[hameau,village, ville, sans], loc="upper right", title="Domicile")

plt.xlabel('pc2')
plt.ylabel('pc3')

plt.savefig('pca-domicile.png',dpi=300)
#plt.show()

