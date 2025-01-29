import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from ML.unsupervised import KMeans


org_img = plt.imread("images/strawberry.png")
print(org_img.shape)
mod_img = org_img.reshape(org_img.shape[0]*org_img.shape[1], 4)

no_of_colors = 5


model = KMeans(K=no_of_colors)
centroid_pos, segregated_points = model.fit(mod_img, max_tries=5, max_iters=5, verbose=True)

compressed_img = centroid_pos[segregated_points]
compressed_img = compressed_img.reshape(org_img.shape[0], org_img.shape[1], 4)

fix, ax = plt.subplots(1, 2, figsize=(10,4))

ax[0].imshow(org_img)
ax[0].set_title("Original Image")
ax[1].imshow(compressed_img)
ax[1].set_title("Compressed Image")

plt.show()