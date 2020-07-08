#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.
import numpy as np
import pylab
import os.path
from PIL import Image
import matplotlib.pyplot as plt

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, output_dir="output_tsne"):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            raise OSError("Can't create destination directory (%s)!" % (output_dir))

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

        if iter%10 ==0:
            pylab.clf()
            pylab.xlim([-120,100])
            pylab.ylim([-100,120])
            pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
            pylab.savefig(os.path.join(output_dir,"tsne_{}.png".format(int(iter/10))))

    # Return solution
    return Y,P,Q

def ssne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, output_dir="output_ssne"):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            raise OSError("Can't create destination directory (%s)!" % (output_dir))

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.                                  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = np.exp(-1. * np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

        if iter%10 ==0:
            pylab.clf()
            pylab.xlim([-15,15])
            pylab.ylim([-15,15])
            pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
            pylab.savefig(os.path.join(output_dir,"ssne_{}.png".format(int(iter/10))))

    # Return solution
    return Y,P,Q

def printSimilarity(P,Q,output_dir):
    pylab.clf()
    plt.subplot(2,1,1)
    pylab.hist(P.flatten(),bins=40,log=True)
    plt.subplot(2,1,2)
    pylab.hist(Q.flatten(),bins=40,log=True)
    plt.savefig(output_dir+"_similarity.png")

def saveGIF(output_dir,mode):
    gifs = []
    if mode == 1:
        for i in range(100):
            gifs.append(Image.open(os.path.join(output_dir,"tsne_{}.png".format(i))))
    else:
        for i in range(100):
            gifs.append(Image.open(os.path.join(output_dir,"ssne_{}.png".format(i))))
    gifs[0].save(output_dir+'.gif', format='GIF',
               append_images=gifs[1:],
               save_all=True,
               duration=300, loop=0)

if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")

    perplexity_list = [10.0, 20.0, 30.0, 40.0, 50.0]
    for perplexity in perplexity_list:
        output_dir="output_tsne"+str(int(perplexity))
        # Y,P,Q = tsne(X, 2, 50, perplexity, output_dir)
        if (os.path.exists('tsne_P_{}.npy'.format(int(perplexity))) and
            os.path.exists('tsne_Q_{}.npy'.format(int(perplexity))) and
            os.path.exists('tsne_Y_{}.npy'.format(int(perplexity)))):
            P = np.load('tsne_P_{}.npy'.format(int(perplexity)))
            Q = np.load('tsne_Q_{}.npy'.format(int(perplexity)))
            Y = np.load('tsne_Y_{}.npy'.format(int(perplexity)))
        else:
            Y,P,Q = tsne(X, 2, 50, perplexity, output_dir)
            np.save('tsne_P_{}.npy'.format(int(perplexity)),P)
            np.save('tsne_Q_{}.npy'.format(int(perplexity)),Q)
            np.save('tsne_Y_{}.npy'.format(int(perplexity)),Y)
        printSimilarity(P,Q,output_dir)
        saveGIF(output_dir,1)

        # pylab.clf()
        # pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
        # pylab.show()

        output_dir="output_ssne"+str(int(perplexity))
        # Y,P,Q = ssne(X, 2, 50, perplexity, output_dir)
        if (os.path.exists('ssne_P_{}.npy'.format(int(perplexity))) and
            os.path.exists('ssne_Q_{}.npy'.format(int(perplexity))) and
            os.path.exists('ssne_Y_{}.npy'.format(int(perplexity)))):
            P = np.load('ssne_P_{}.npy'.format(int(perplexity)))
            Q = np.load('ssne_Q_{}.npy'.format(int(perplexity)))
            Y = np.load('ssne_Y_{}.npy'.format(int(perplexity)))
        else:
            Y,P,Q = ssne(X, 2, 50, perplexity, output_dir)
            np.save('ssne_P_{}.npy'.format(int(perplexity)),P)
            np.save('ssne_Q_{}.npy'.format(int(perplexity)),Q)
            np.save('ssne_Y_{}.npy'.format(int(perplexity)),Y)
        printSimilarity(P,Q,output_dir)
        saveGIF(output_dir,2)
