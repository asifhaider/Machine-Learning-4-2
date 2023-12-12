from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def low_rank_approximation(A, k):
    # Perform Singular Value Decomposition
    U, S, VT = np.linalg.svd(A)
    # Construct the diagonal matrix of singular values
    S_k = np.diag(S[:k])
    # Construct the matrix of k singular values
    U_k = U[:, :k]
    # Construct the matrix of k singular values
    VT_k = VT[:k, :]
    # Reconstruct the matrix A from the k singular values
    A_k = np.dot(np.dot(U_k, S_k), VT_k)
    return A_k


def main():
    # Read image
    img = Image.open('image.jpg')

    # Convert image to grayscale
    gray_img = img.convert('L')

    # Convert image to numpy array
    original_array = np.array(img)
    gray_array = np.array(gray_img)

    # # Display the original and grayscale image
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_array)
    # plt.title('Original Image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(gray_array, cmap='gray')
    # plt.title('Grayscale Image')
    # # plt.show()

    print(original_array.shape)
    print(gray_array.shape)

    # # Perform Singular Value Decomposition
    # U, S, VT = np.linalg.svd(gray_array)

    # Vary the value of k from 1 to min(n, m) (taking at least 10 such values in the interval)
    # k_values = np.linspace(1, min(gray_array.shape), 10, dtype=int)
    k_values = [1, 5, 10, 20, 25, 30, 35, 50, 100, 500]

    # Plot the resultant k-rank approximation as a grayscale image
    plt.figure(figsize=(20, 10))

    for i, k in enumerate(k_values):
        # Reconstruct the matrix A from the k singular values
        A_k = low_rank_approximation(gray_array, k)
        plt.subplot(2, 5, i + 1)
        plt.imshow(A_k, cmap='gray')
        plt.title(f'Rank {k} Approximation')

    plt.show()


if __name__ == '__main__':
    main()