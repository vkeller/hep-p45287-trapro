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

CustomCmap = ListedColormap(['green','red','blue'])
scatter = ax.scatter(x[:,1],x[:,2],c=donnees['c_sexe'],cmap=CustomCmap)

sans = mpatches.Patch(color='g', label='Pas de réponse')
femmes = mpatches.Patch(color='r', label='Femmes')
hommes = mpatches.Patch(color='b', label='Hommes')

plt.legend(handles=[femmes,hommes,sans], loc="upper right", title="Sexe")

plt.xlabel('pc2')
plt.ylabel('pc3')

plt.savefig('pca-sexe.png',dpi=300)
#plt.show()

