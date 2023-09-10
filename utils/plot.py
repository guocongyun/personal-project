import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
# Setting the seed for reproducibility
np.random.seed(42)

import numpy as np
from sklearn.preprocessing import MinMaxScaler



fig, axes = plt.subplots(2, 3, figsize=(20, 10))

k_values = [0, 0.05, 0.5]
conf = Z_confidence.reshape(xx.shape)
# This function adjusts the confidence using neighbors and displays the plot
labels = ["Class A", "Class B", "Class C", "Class D"]
def adjust_confidence_and_plot(ax_row, k, data_multiplier=1):
    for ids, k in enumerate(k_values):
        adjusted_confidences = []
        for idx, point in enumerate(np.c_[xx.ravel(), yy.ravel()]):
            if data_multiplier: 
                query_distance_matrix = cosine_distances(point.reshape(1, -1), X)
                distances, indices = nbrs.kneighbors(query_distance_matrix)
            else: 
                query_distance_matrix = cosine_distances(point.reshape(1, -1), Xinf)
                distances, indices = nbrsinf.kneighbors(query_distance_matrix)

            adjusted_confidence = conf[idx//100][idx%100]*(1-k)
            for i, index in enumerate(indices[0]):
                sim = 1 - distances[0][i]
                if data_multiplier:
                    adjusted_confidence += k*sim * 1*0.1
                else:
                    adjusted_confidence += k*sim * data_conf_inf[index]*0.1
                    
            adjusted_confidences.append(adjusted_confidence)
        
        adjusted_confidences = np.array(adjusted_confidences)
        # adjusted_confidences = (adjusted_confidences - adjusted_confidences.min()) / (adjusted_confidences.max() - adjusted_confidences.min())
    
        contour = ax_row[ids].contourf(-xx, -yy, adjusted_confidences.reshape(xx.shape), levels=np.linspace(0, 1, 11), cmap="Greys_r", alpha=0.8)
        for class_idx, label in enumerate(labels):
            mask = yinf == class_idx
            ax_row[ids].scatter(-Xinf[mask, 0], -Xinf[mask, 1],c=yinf[mask], edgecolors='k', marker='o', s=50, cmap="Greys_r", vmin=0, vmax=3, label=label)
        # else: ax_row[ids].scatter(-Xinf[:, 0], -Xinf[:, 1], c=yinf, edgecolors='k', marker='o', s=50, cmap="Greys_r", vmin=0, vmax=3)
        if not data_multiplier:
            ax_row[ids].set_title(f'w={k}', fontsize=20)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])  # Adjust position & dimensions as needed
    cb = fig.colorbar(contour, cax=cbar_ax)
    cb.set_label('Confidence', fontsize=20)
    # Set title for the row
    ax_row[0].set_ylabel("KNN ID" if data_multiplier else "KNN TD", fontsize=20)
# Adjusting confidence and plotting for both approaches (data_conf multiplier and 1)
adjust_confidence_and_plot(axes[0], 1, data_multiplier=0)
adjust_confidence_and_plot(axes[1], 1, data_multiplier=1)
axes[0][0].legend(fontsize=20)
plt.tight_layout(rect=[0, 0, 0.91, 1])
plt.savefig('4.4.3.png')  # Choose a suitable name and format for your figure
plt.show()
