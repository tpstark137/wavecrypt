import cv2
import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x0, size):
    x = np.zeros(size, dtype=np.float32)
    x[0] = x0
    for i in range(1, size):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x

def generate_chaotic_sequence(shape, r=3.99, x0=0.5):
    total_size = np.prod(shape)
    chaotic_seq = logistic_map(r, x0, total_size)
    chaotic_seq = np.floor(chaotic_seq * 256).astype(np.uint8)
    return chaotic_seq.reshape(shape)

def encrypt_image(img, chaotic_seq):
    encrypted = cv2.bitwise_xor(img, chaotic_seq)
    return encrypted

def decrypt_image(encrypted_img, chaotic_seq):
    decrypted = cv2.bitwise_xor(encrypted_img, chaotic_seq)
    return decrypted

def main():
    # Load the image in grayscale for simplicity
    img = cv2.imread('test_image.jpeg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found. Please place 'sample_image.jpg' in the script directory.")
        return

    # Generate chaotic key stream
    chaotic_seq = generate_chaotic_sequence(img.shape, r=3.99, x0=0.4)

    # Encrypt and Decrypt
    encrypted_img = encrypt_image(img, chaotic_seq)
    #chaotic_seq = generate_chaotic_sequence(img.shape, r=3, x0=0.4)
    decrypted_img = decrypt_image(encrypted_img, chaotic_seq)

    # Display results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Encrypted Image")
    plt.imshow(encrypted_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Decrypted Image")
    plt.imshow(decrypted_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
