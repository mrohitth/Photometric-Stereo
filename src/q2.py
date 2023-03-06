# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 12, 2022
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimateShape, plotSurface 
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface 

import cv2

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    B = None
    L = None
    # Your code here
    u, s, v = np.linalg.svd(I, full_matrices=False)
    s[3:] = 0.

    S31 = np.diag(s[:3])
    VT31 = v[:3,:]
    B = np.dot(np.sqrt(S31),VT31)
    L = np.dot(u[:,:3],np.sqrt(S31)).T
    return B, L

def plotBasRelief(B, mu, nu, lam):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter
    
    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    # Your code here
    I, originalL, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)

    pass

if __name__ == "__main__":

    I, L0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)

    mu = 10  # -0.5, 0.5, 1
    nu = 0  # 0.5, 1, 3
    lam = 0.1  # -1, 1.5, 3
    G = np.asarray([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    G_invT=np.linalg.inv(G).T
    Nt = enforceIntegrability(B, s)
    GTB=np.dot(G_invT,Nt)

    albedos, normals = estimateAlbedosNormals(B)
    surface = estimateShape(normals, s)

    min_v, max_v = np.min(surface), np.max(surface)
    surface = (surface - min_v) / (max_v - min_v)

    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)


    # Albedos Image
    plt.imshow(albedoIm, cmap='gray')
    cv2.imwrite('q2b_albedo.png', (albedoIm*255))
    plt.show()


    # Normals Image
    # normalIm = normalize(normalIm)
    plt.imshow(normalIm, cmap='rainbow')
    plt.savefig('q2b_normal.png')
    plt.show()

    # Q2.d
    
    surface = (surface * 255.).astype('uint8')
    plotSurface(surface)
