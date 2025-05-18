import numpy as np
import pywt
import cv2
from PIL import Image

def arnold_transform(img, iterations):
    N = img.shape[0]
    scrambled_img = np.copy(img)
    for _ in range(iterations):
        temp_img = np.copy(scrambled_img)
        for x in range(N):
            for y in range(N):
                x_new = (x + y) % N
                y_new = (x + 2 * y) % N
                scrambled_img[x_new, y_new] = temp_img[x, y]
    return scrambled_img

def inverse_arnold_transform(img, iterations):
    N = img.shape[0]
    descrambled_img = np.copy(img)
    for _ in range(iterations):
        temp_img = np.copy(descrambled_img)
        for x in range(N):
            for y in range(N):
                x_new = (2 * x - y) % N
                y_new = (-x + y) % N
                descrambled_img[x_new, y_new] = temp_img[x, y]
    return descrambled_img

def logistic_map(size, key):
    r = 3.99
    x = key
    chaos_seq = np.zeros(size)
    for i in range(size):
        x = r * x * (1 - x)
        chaos_seq[i] = x
    return (chaos_seq * 255).astype(np.uint8)

def encrypt_image(image_path, key, iterations):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    scrambled = arnold_transform(img, iterations)

    LL, (LH, HL, HH) = pywt.dwt2(scrambled.astype(np.float64), 'db4')

    chaos_seq = logistic_map(rows * cols, key).reshape(rows, cols)
    LL_encrypted = np.bitwise_xor(LL.astype(np.uint8), chaos_seq[:LL.shape[0], :LL.shape[1]])

    encrypted_img = pywt.idwt2((LL_encrypted.astype(np.float64), (LH, HL, HH)), 'db4')
    encrypted_img = np.clip(encrypted_img, 0, 255).astype(np.uint8)

    cv2.imwrite('encrypted_image.png', encrypted_img)
    return encrypted_img

def decrypt_image(encrypted_img, key, iterations):
    if isinstance(encrypted_img, str):
        encrypted_img = cv2.imread(encrypted_img, cv2.IMREAD_GRAYSCALE)

    rows, cols = encrypted_img.shape
    LL_encrypted, (LH, HL, HH) = pywt.dwt2(encrypted_img.astype(np.float64), 'db4')

    chaos_seq = logistic_map(rows * cols, key).reshape(rows, cols)
    LL_decrypted = np.bitwise_xor(LL_encrypted.astype(np.uint8), chaos_seq[:LL_encrypted.shape[0], :LL_encrypted.shape[1]])

    scrambled = pywt.idwt2((LL_decrypted.astype(np.float64), (LH, HL, HH)), 'db4')
    scrambled = np.clip(scrambled, 0, 255).astype(np.uint8)

    decrypted = inverse_arnold_transform(scrambled, iterations)

    cv2.imwrite('decrypted_image.png', decrypted)
    return decrypted

# ----------- Testing the Encryption-Decryption Process ----------- #
if __name__ == "__main__":
    image_path = 'test_image.jpeg'  # Replace with your image
    key = 0.618
    iterations = 10

    encrypted_img = encrypt_image(image_path, key, iterations)
    cv2.imshow('Encrypted Image', encrypted_img)
    cv2.waitKey(0)

    decrypted_img = decrypt_image('encrypted_image.png', key, iterations)
    cv2.imshow('Decrypted Image', decrypted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
