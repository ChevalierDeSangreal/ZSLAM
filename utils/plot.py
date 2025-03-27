import torch
import matplotlib.pyplot as plt

def bool_tensor_visualization(tensor):
    plt.imshow(tensor, cmap='gray')
    plt.show()