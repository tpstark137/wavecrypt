import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os

# Logistic Map Function for Chaotic Sequence Generation
def logistic_map(size, key):
    x = key
    r = 3.99
    sequence = np.zeros(size, dtype=np.float64)

    for i in range(size):
        x = r * x * (1 - x)
        sequence[i] = x

    return np.uint8(sequence * 255)

# Encrypt Image using Wavelet Transform & Chaos
def encrypt_image(image_path, key):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    rows, cols = img.shape

    coeffs = pywt.dwt2(img, 'db4')
    LL, (LH, HL, HH) = coeffs

    chaos_seq = logistic_map(LL.size, key).reshape(LL.shape)
    LL_encrypted = np.bitwise_xor(LL.astype(np.uint8), chaos_seq)

    encrypted_img = pywt.idwt2((LL_encrypted, (LH, HL, HH)), 'db4')
    encrypted_img = np.clip(encrypted_img, 0, 255).astype(np.uint8)

    return encrypted_img

# Decrypt Image using Wavelet Transform & Chaos
def decrypt_image(encrypted_img, key):
    rows, cols = encrypted_img.shape

    coeffs = pywt.dwt2(encrypted_img, 'db4')
    LL_encrypted, (LH, HL, HH) = coeffs

    chaos_seq = logistic_map(LL_encrypted.size, key).reshape(LL_encrypted.shape)
    LL_decrypted = np.bitwise_xor(LL_encrypted.astype(np.uint8), chaos_seq)

    decrypted_img = pywt.idwt2((LL_decrypted, (LH, HL, HH)), 'db4')
    decrypted_img = np.clip(decrypted_img, 0, 255).astype(np.uint8)

    return decrypted_img

# --- Main Flow ---
image_path = 'test_image.jpeg'
key = 0.618  # Between 0 and 1

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Make sure the image '{image_path}' exists in this directory: {os.getcwd()}")

# Encrypt and Save
encrypted_img = encrypt_image(image_path, key)
cv2.imwrite("encrypted_image.jpg", encrypted_img)

# Decrypt and Save
decrypted_img = decrypt_image(encrypted_img, key)
cv2.imwrite("decrypted_image.jpg", decrypted_img)

# Display
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Encrypted")
plt.imshow(encrypted_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Decrypted")
plt.imshow(decrypted_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
