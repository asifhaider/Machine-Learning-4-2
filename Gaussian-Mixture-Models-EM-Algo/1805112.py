"""
Dimensionality Reduction using Singular Value Decomposition (SVD) and Principal Component Analysis (PCA) manually 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvn
import argparse

np.random.seed(42)

"""
Input:
    D: Input data matrix of shape (N, M) where N is the number of samples and M is the number of features
Output:
    projected data matrix along the two most prominent principal components/ axes
"""

def data_preprocess(D):
    # z-normalize: (data-mean)/standard deviation
    D = (D - np.mean(D, axis=0))/np.std(D, axis=0)
    return D


def principal_component_analysis(D):
    # if the feature dimension is greater than 2, then perform PCA only
    if D.shape[1] <= 2:
        return D
    
    # perform SVD on the data matrix
    U, S, Vt = np.linalg.svd(D, full_matrices=False)

    # here, U is the left singular vector matrix of shape (N, M), indicating the direction of maximum variance in original feature space
    # S is the diagonal matrix of singular values of shape (M, M), indicating the importance of each principal component
    # Vt is the right singular vector matrix of shape (M, M), indicating the direction of maximum variance in the projected feature space, or the direction of maximum variance in the original feature space corresponding to each principal component
    
    # project the data onto the two most prominent principal components
    principal_axes = Vt[:2, :]
    D = np.dot(D, principal_axes.T) # because already transposed
    return D


"""
Plot the data points in the 2D space after dimensionality reduction, using seaborn library. Then save the plot.
"""

def plot_data(D, path_to_save):
    sns.set(style='darkgrid')
    # plot the data points in the 2D space after dimensionality reduction
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=D[:, 0], y=D[:, 1], c='b', marker='o', label='Data points')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best')
    plt.title('Data points in the 2D space')
    plt.grid(True)
    plt.savefig(path_to_save)
    plt.show()


"""
Parameter initialization for the EM algorithm
"""

def initialize_parameters(D, K):
    samples, features = D.shape

    # initialize the mean vectors randomly
    means = D[np.random.choice(samples, K, replace=False)]

    # initialize the covariance matrices as identity matrices
    covariances = np.tile(np.identity(features), (K, 1, 1))

    # initialize the weights uniformly
    weights = np.ones(K)/K

    return means, covariances, weights


"""
Expectation step of the EM algorithm
"""

def expectation_step(D, means, covariances, weights):
    samples, features = D.shape
    K = weights.shape[0]

    conditional_probabilities = np.zeros((samples, K))

    for k in range(K):
        # calculate multi-variate normal probability density function
        mvn_pdf = mvn.pdf(D, mean=means[k], cov=covariances[k])

        # calculate the conditional probabilities
        conditional_probabilities[:, k] = weights[k] * mvn_pdf

    # normalize the conditional probabilities
    conditional_probabilities /= np.sum(conditional_probabilities, axis=1, keepdims=True)

    return conditional_probabilities 
    

"""
Maximization step of the EM algorithm
"""

def maximization_step(D, conditional_probabilities):
    samples, features = D.shape
    K = conditional_probabilities.shape[1]

    # update the means
    means = np.dot(conditional_probabilities.T, D)/np.sum(conditional_probabilities, axis=0, keepdims=True).T

    # update the covariances
    covariances = np.zeros((K, features, features))
    for k in range(K):
        # calculate the difference between the data points and the mean
        difference = D - means[k]
        weighted_difference = np.dot(conditional_probabilities[:, k] * difference.T, difference)
        covariances[k] = weighted_difference/np.sum(conditional_probabilities[:, k])

    # update the weights
    weights = np.sum(conditional_probabilities, axis=0)/samples

    return means, covariances, weights


"""
Calculate the log-likelihood of the data
"""

def calculate_log_likelihood(D, means, covariances, weights):
    K = weights.shape[0]

    # likelihood matrix, probability of each data point belonging to each component, size nxk
    likelihood = np.zeros((len(D), K))
    for k in range(K):
        # add a small value to the diagonal of the covariance matrix to avoid singular matrix: known as regularization
        covariances[k] += 1e-6 * np.eye(covariances[k].shape[0])
        likelihood[:, k] = weights[k] * mvn.pdf(D, means[k], covariances[k])

    # calculate the log-likelihood
    log_likelihood = np.sum(np.log(np.sum(likelihood, axis=1)))
    return log_likelihood



""" 
EM algorithm for GMM
"""

def EM_algorithm(D, K, max_iterations, threshold):
    # initialize the parameters
    means, covariances, weights = initialize_parameters(D, K)
    log_likelihoods = []

    for _ in range(max_iterations):
        # E-step
        conditional_probabilities = expectation_step(D, means, covariances, weights)

        # M-step
        means, covariances, weights = maximization_step(D, conditional_probabilities)

        # calculate the log-likelihood
        log_likelihood_value = calculate_log_likelihood(D, means, covariances, weights)
        log_likelihoods.append(log_likelihood_value)

        # check for convergence
        if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < threshold:
            break

    return means, covariances, weights, log_likelihoods



""" 
Plot convergence of the log-likelihood vs components for each value of component K. Use seaborn darkgrid style for plotting.
"""

def plot_log_likelihood(log_likelihoods, k_ranges, path_to_save):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 10))
    sns.lineplot(x=k_ranges, y=log_likelihoods, marker='o', color='blue')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('Convergence Log-likelihood')
    plt.title('Convergence of the log-likelihood vs components')
    plt.grid(True)
    plt.savefig(path_to_save)
    plt.show()


"""
Plot the estimated GMM components. Use seaborn library for plotting.
"""

def plot_gmms(D, means, covariances, K, path_to_save):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 10))
    # sns.scatterplot(x=D[:, 0], y=D[:, 1], c='b', marker='o', label='Data points')
    
    # Plot the estimated GMM components
    for k in range(K):
        x, y = np.random.multivariate_normal(means[k], covariances[k], 500).T
        sns.scatterplot(x=x, y=y, label=f'Gaussian {k+1}')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best')
    plt.title(f'Estimated GMM components for K = {K}')
    plt.grid(True)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()



if __name__ == '__main__':

    """
    Read the dataset and perform PCA
    """

    parser = argparse.ArgumentParser("Provide the path to the dataset")
    parser.add_argument("path", type=str, help="Path to the dataset")
    parser.add_argument("lower_bound", type=int, help="Lower bound of the range of components")
    parser.add_argument("upper_bound", type=int, help="Upper bound of the range of components")
    args = parser.parse_args()

    # data_path = 'Dataset/6D_data_points.txt'
    data_path = args.path
    reduced_image_path = data_path.split('/')[1].split('.')[0] + '.png'
    gmm_image_path = data_path.split('/')[1].split('.')[0] + '_GMM_'
    log_likelihood_image_path = data_path.split('/')[1].split('.')[0] + '_Log_likelihood.png'
    

    # read the dataset
    D = pd.read_csv(data_path, header=None).values
    print('Shape of the data matrix before PCA:', D.shape)

    # perform data preprocessing
    D = data_preprocess(D)

    # perform PCA
    D = principal_component_analysis(D)
    print('Shape of the data matrix after PCA:', D.shape)

    # plot the data points in the 2D space after dimensionality reduction
    plot_data(D, reduced_image_path)

    """
    Expectation-Maximization (EM) algorithm for Gaussian Mixture Model (GMM)
    """
    
    k_range = range(args.lower_bound, args.upper_bound+1)
    best_log_likelihoods = []
    gmm_parameters_list = []
    best_k = None

    for k in k_range:
        best_log_likelihood = float('-inf')

        # run the EM algorithm for 5 different initializations
        for _ in range(5):
            # run the EM algorithm
            means, covariances, weights, log_likelihoods = EM_algorithm(D, k, max_iterations=1000, threshold=1e-8)
            log_likelihood = max(log_likelihoods)
            print(f'K = {k}, Log-likelihood = {log_likelihood}')
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_parameters = (means, covariances, weights)

        best_log_likelihoods.append(best_log_likelihood)
        print(f'Best Log-likelihood = {best_log_likelihood}')
        gmm_parameters_list.append(best_parameters)

        # plot estimated GMM components
        plot_gmms(D, best_parameters[0], best_parameters[1], k, gmm_image_path+str(k)+'.png')   

    # plot convergence of the log-likelihood vs components for each value of component K
    plot_log_likelihood(best_log_likelihoods, k_range, log_likelihood_image_path) 
