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

CustomCmap = ListedColormap(['cyan','lightgray','silver','darkgrey','darkgray', 'grey','gray','dimgrey','dimgray','black'])
scatter = ax.scatter(x[:,1],x[:,2],c=donnees['c_age'],cmap=CustomCmap)

sans = mpatches.Patch(color='cyan', label='Pas de réponse')
c1 = mpatches.Patch(color='lightgray', label='-30 ans')
c2 = mpatches.Patch(color='silver', label='30-35 ans')
c3 = mpatches.Patch(color='darkgrey', label='35-40 ans')
c4 = mpatches.Patch(color='darkgray', label='40-45 ans')
c5 = mpatches.Patch(color='grey', label='45-50 ans')
c6 = mpatches.Patch(color='gray', label='50-55 ans')
c7 = mpatches.Patch(color='dimgrey', label='55-60 ans')
c8 = mpatches.Patch(color='dimgray', label='60-65 ans')
c9 = mpatches.Patch(color='black', label='+65 ans')


plt.legend(handles=[c1, c2, c3, c4, c5, c6, c7, c8, c9, sans], loc="upper right", title="Classe d'âge'")

plt.xlabel('pc2')
plt.ylabel('pc3')

plt.savefig('pca-age.png',dpi=300)
#plt.show()

