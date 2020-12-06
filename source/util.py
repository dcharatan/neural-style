import torch
import numpy as np
import matplotlib.pyplot as plt

# Computes the Gram matrix given phi_j(x)
#  "The Gram matrix can be computed efficiently by reshaping phi_j(x) into a matrix psi of
# shape C_j × H_jW_j ; then G^phi_j(x) = psi psi^T / C_jH_jW_j"
# https://arxiv.org/pdf/1603.08155.pdf
def gram_matrix(x):
    (j, c, h, w) = x.size()
    psi = x.view(j, c, h * w)
    psi_T = torch.transpose(psi, 1, 2)
    G = psi.bmm(psi_T) / (c * h * w)
    return G


def show_img(data):
    # Revert the normalization based on pretrained torchvision models
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = (img * std + mean).transpose(1, 2, 0)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    plt.imshow(img)
    plt.show()