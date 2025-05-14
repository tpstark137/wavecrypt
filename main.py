import numpy as np
import cv2
import pywt
from scipy.linalg import svd


# Arnold Cat Map Transformation
def arnold_cat_map(img, iterations=5):
    h, w = img.shape
    transformed = np.copy(img)
    for _ in range(iterations):
        temp = np.zeros_like(transformed)
        for x in range(h):
            for y in range(w):
                new_x = (x + y) % h
                new_y = (x + 2 * y) % w
                temp[new_x, new_y] = transformed[x, y]
        transformed = temp
    return transformed


# Inverse Arnold Cat Map
def inverse_arnold_cat_map(img, iterations=5):
    h, w = img.shape
    transformed = np.copy(img)
    for _ in range(iterations):
        temp = np.zeros_like(transformed)
        for x in range(h):
            for y in range(w):
                new_x = (2 * x - y) % h
                new_y = (-x + y) % w
                temp[new_x, new_y] = transformed[x, y]
        transformed = temp
    return transformed


# Logistic Map for Chaotic Sequence
def logistic_map(size, r=3.9, x0=0.5):
    x = np.zeros(size)
    x[0] = x0
    for i in range(1, size):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x  # Keep floating-point values


# Encrypt Image using DWT + SVD + ACM + Chaos
def encrypt_image(img_path):
    # Read Image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Convert to Grayscale for Processing
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # Apply DWT using "db4"
    coeffs = pywt.dwt2(gray, 'db4')
    LL, (LH, HL, HH) = coeffs

    # Save DWT Processed Image (Reconstruct LL band)
    dwt_image = pywt.idwt2((LL, (np.zeros_like(LH), np.zeros_like(HL), np.zeros_like(HH))), 'db4')
    cv2.imwrite("dwt_image.jpg", np.clip(dwt_image, 0, 255).astype(np.uint8))

    # Apply SVD on LL Sub-band
    U, S, Vt = svd(LL)

    # Reconstruct LL using SVD
    svd_image = np.dot(U, np.dot(np.diag(S), Vt))
    cv2.imwrite("svd_image.jpg", np.clip(svd_image, 0, 255).astype(np.uint8))

    # Generate Chaotic Sequence
    chaotic_seq = logistic_map(len(S))

    # Encrypt Singular Values
    S_encrypted = S + chaotic_seq

    # Save Chaos Applied Image
    chaos_image = np.dot(U, np.dot(np.diag(S_encrypted), Vt))
    cv2.imwrite("chaos_image.jpg", np.clip(chaos_image, 0, 255).astype(np.uint8))

    # Apply Arnold Cat Map on U and V matrices
    U_encrypted = arnold_cat_map(U)
    Vt_encrypted = arnold_cat_map(Vt)

    # Save Arnold Transformed Image (Reconstructed LL after Arnold Map)
    arnold_image = np.dot(U_encrypted, np.dot(np.diag(S_encrypted), Vt_encrypted))
    cv2.imwrite("arnold_image.jpg", np.clip(arnold_image, 0, 255).astype(np.uint8))

    # Reconstruct Encrypted LL
    LL_encrypted = np.dot(U_encrypted, np.dot(np.diag(S_encrypted), Vt_encrypted))

    # Apply Inverse DWT to Get Encrypted Image
    encrypted_img = pywt.idwt2((LL_encrypted, (LH, HL, HH)), 'db4')

    return encrypted_img, U_encrypted, S_encrypted, Vt_encrypted, chaotic_seq, LH, HL, HH


# Decrypt Image
def decrypt_image(U, S, Vt, chaotic_seq, LH, HL, HH):
    # Recover Singular Values
    S_decrypted = S - chaotic_seq

    # Apply Inverse Arnold Cat Map to Recover U and V
    U_decrypted = inverse_arnold_cat_map(U, iterations=5)
    Vt_decrypted = inverse_arnold_cat_map(Vt, iterations=5)

    # Reconstruct Decrypted LL Sub-band
    LL_decrypted = np.dot(U_decrypted, np.dot(np.diag(S_decrypted), Vt_decrypted))

    # Apply Inverse DWT
    decrypted_img = pywt.idwt2((LL_decrypted, (LH, HL, HH)), 'db4')
    
    return np.clip(decrypted_img, 0, 255)  # Ensure valid pixel values


# Example Usage
image_path = "input_image.jpg"
encrypted_img, U, S, Vt, chaotic_seq, LH, HL, HH = encrypt_image(image_path)

# Save Encrypted Image
cv2.imwrite("encrypted_image.jpg", np.clip(encrypted_img, 0, 255).astype(np.uint8))

# Decrypt
decrypted_img = decrypt_image(U, S, Vt, chaotic_seq, LH, HL, HH)

# Save Decrypted Image
cv2.imwrite("decrypted_image.jpg", decrypted_img.astype(np.uint8))
