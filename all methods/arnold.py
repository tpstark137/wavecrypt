import numpy as np
import cv2
import matplotlib.pyplot as plt

def arnold_transform(img, iterations):
    """Applies the Arnold transform (scrambling) to a square image."""
    N = img.shape[0]
    scrambled_img = img.copy()
    for _ in range(iterations):
        temp_img = scrambled_img.copy()
        for x in range(N):
            for y in range(N):
                x_new = (x + y) % N
                y_new = (x + 2 * y) % N
                scrambled_img[x_new, y_new] = temp_img[x, y]
    return scrambled_img

def inverse_arnold_transform(img, iterations):
    """Applies the inverse Arnold transform (descrambling) to a square image."""
    N = img.shape[0]
    descrambled_img = img.copy()
    for _ in range(iterations):
        temp_img = descrambled_img.copy()
        for x in range(N):
            for y in range(N):
                x_new = (2 * x - y) % N
                y_new = (-x + y) % N
                descrambled_img[x_new, y_new] = temp_img[x, y]
    return descrambled_img

def main():
    # Load and preprocess image
    img = cv2.imread('test_image.jpeg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found. Make sure 'test_image.jpg' is in the current directory.")

    # Resize to square
    N = min(img.shape[:2])
    img = cv2.resize(img, (N, N))

    iterations = 10

    # Apply Arnold Transform
    scrambled = arnold_transform(img, iterations)

    # Apply Inverse Arnold Transform
    descrambled = inverse_arnold_transform(scrambled, iterations)

    # Display Results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Scrambled ({iterations} iters)")
    plt.imshow(scrambled, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Descrambled")
    plt.imshow(descrambled, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
