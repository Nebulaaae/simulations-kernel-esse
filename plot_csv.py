import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np

import matplotlib.pyplot as plt

# Importer le CSV
df = pd.read_csv('output/kernel_accumulated.csv')

# Créer la figure
plt.figure(figsize=(10, 8))

# Si vous avez beaucoup de points au même endroit, utiliser la densité
if len(df) > 0:
    # Compter les points à chaque position
    points = df.groupby(['PostPosition_X', 'PostPosition_Y']).size().reset_index(name='count')
    
    # Scatter plot avec la taille/couleur basée sur le nombre de points
    scatter = plt.scatter(points['PostPosition_X'], points['PostPosition_Y'], s=points['count']*50, 
                         c=points['count'], cmap='YlOrRd', alpha=0.6, edgecolors='black')
    
    plt.colorbar(scatter, label='Nombre de points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Distribution des points - Intensité par densité')
    plt.grid(True, alpha=0.3)
    plt.show()