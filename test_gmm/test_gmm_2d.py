import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# Generate sample data
X = np.array([[1, 2], [1, 4], [2, 1], [10, 2], [10, 4], [12, 2]])

# Fit a Gaussian Mixture Model
gm = GaussianMixture(n_components=2, random_state=0).fit(X)

# Function to draw ellipse
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ell = Ellipse(xy=position, width=nsig * width, height=nsig * height, angle=angle, **kwargs)
        ax.add_patch(ell)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Data points')
plt.scatter(gm.means_[:, 0], gm.means_[:, 1], c='red', marker='x', s=100, label='Centers')

for pos, covar in zip(gm.means_, gm.covariances_):
    draw_ellipse(pos, covar, alpha=0.2, color='red')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Mixture Model with Covariance Ellipses')
plt.legend()
plt.show()