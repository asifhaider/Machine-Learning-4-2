# Author: Md. Asif Haider, 1805112

import numpy as np

def produce_random_invertible_matrix(n):
    failed = 0 
    total = 0
    # Produces a random n x n invertible matrix A. For the purpose of demonstrating, every cell of A will be an integer between -100 and 100
    while True:
        A = np.random.randint(-10, 10, (n, n))
        total += 1
        # Check if the matrix A is invertible by checking if the determinant of A is not equal to 0
        if np.linalg.det(A) != 0:
            return A, (total-failed)/total*100
        else:
            failed += 1


def reconstruct_matrix_from_eigenvalues_and_eigenvectors(eigenvalues, eigenvectors):
    # Why the order matters? Because the eigenvectors are the columns of the matrix eigenvectors
    # And the eigenvalues are the diagonal elements of the matrix eigenvalues
    # So right multiplying the eigenvectors matrix with the eigenvalues diagonal matrix will give us the reconstructed matrix
    A = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    return A


def main():

    # Taking the dimensions of matrix n as input
    n = int(input("Enter the dimensions of matrix (n): "))

    # Producing a random n x n invertible matrix A
    A, success = produce_random_invertible_matrix(n)

    # Print the matrix A and the success rate of producing an invertible matrix in 2 decimal places
    print("The matrix A is: \n")
    print(A)
    print("\nThe success rate of producing an invertible matrix is: {:.2f}%\n".format(success))

    # Perform Eigen Decomposition using NumPy’s library function
    eigenvalues, eigenvectors = np.linalg.eig(A)

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
        print("The reconstruction worked properly (taking real values and rounding the fractions)!")
    else:
        print("The reconstruction failed!")


if __name__ == "__main__":
    main()