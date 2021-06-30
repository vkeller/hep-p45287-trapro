
# import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#input_file="resultats.csv"
input_file="resultats-epures.csv"
df1 = pd.read_csv(input_file, header = 0)

# Scale data befor applying PCA
scaling=StandardScaler()
  
# Use fit and transform method 
scaling.fit(df1)
Scaled_data=scaling.transform(df1)
  
# Set the n_components=3
principal=PCA(n_components=3)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)
  
# Check the dimensions of data after PCA
print(x.shape)

# Check the values of eigen vectors
# prodeced by principal components
print(principal.components_[0])

plt.figure(figsize=(10,10))
donnees = pd.read_csv("donnees.csv", header = 0)

#plt.scatter(x[:,0],x[:,1],c=data['target'],cmap='plasma')
# 2 composantes principales
#plt.scatter(x[:,0],x[:,1],c=donnees['c_sexe'],cmap='plasma')
plt.scatter(x[:,0],x[:,2],c=donnees['parti'],cmap='plasma')
cbar = plt.colorbar(label='position politique', ticks = [3,6,9])
cbar.ax.set_yticklabels(['gauche', 'centre', 'droite'])

#plt.scatter(x[:,0],x[:,1],c=donnees['c_age'],cmap='plasma')
#plt.scatter(x[:,0],x[:,1],c=donnees['c_habitat'],cmap='plasma')

plt.xlabel('pc1')
plt.ylabel('pc3')
#plt.show()
plt.savefig('pca-partis.png',dpi=300)

