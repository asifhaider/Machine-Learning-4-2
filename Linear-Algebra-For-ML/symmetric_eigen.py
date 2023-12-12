# Author: Md. Asif Haider, 1805112

import numpy as np

def produce_random_symmetric_matrix(n):
    # Produces a random n x n symmetric matrix A. For the purpose of demonstrating, every cell of A will be an integer between -100 and 100
    A = np.random.randint(-10, 10, (n, n))
    # Make the matrix symmetric by adding it to its transpose
    A = A + A.T
    return ensure_strict_diagonal_dominance(A)


def ensure_strict_diagonal_dominance(A):
    # Make the matrix diagonally dominant by adding the absolute value of the largest diagonal element to all the diagonal elements
    # This will ensure that the matrix is invertible
    A = A + np.diag(np.abs(A).max() * 2 * np.ones(A.shape[0]))
    return A


def reconstruct_matrix_from_eigenvalues_and_eigenvectors(eigenvalues, eigenvectors):
    # Why the transpose works here instead of the inverse?
    # Because symmetric matrices are always diagonalizable and their eigenvectors are orthogonal
    A = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), eigenvectors.T)
    return A


def main():

    # Taking the dimensions of matrix n as input
    n = int(input("Enter the dimensions of matrix (n): "))

    # Producing a random n x n invertible symmetric matrix A
    A = produce_random_symmetric_matrix(n)

    # Print the matrix A
    print("The matrix A is: \n")
    print(A)

    # Perform Eigen Decomposition using NumPy’s library function
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    print("The eigenvalues of A are: \n")
    print(eigenvalues)

    print("\nThe eigenvectors of A are: \n")
    print(eigenvectors)

    # Reconstruct A from eigenvalues and eigenvectors
    A_reconstructed = reconstruct_matrix_from_eigenvalues_and_eigenvectors(eigenvalues, eigenvectors)

    print("\nThe reconstructed matrix A is: \n")
    print(A_reconstructed)

    # Check if the reconstruction worked properly
    print("\nChecking if the reconstruction worked properly (using numpy.allclose()): \n")
    if np.allclose(A, A_reconstructed):
        print("The reconstruction worked properly (rounding the fractions)!")
    else:
        print("The reconstruction failed!")


if __name__ == "__main__":
    main()