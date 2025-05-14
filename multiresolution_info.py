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
    return x


# Function to Apply 3-Level DWT and Save Images
def apply_dwt_and_save(img_gray):
    coeffs = pywt.wavedec2(img_gray, 'db4', level=3)  # 3-Level DWT decomposition
    LL1, (LH1, HL1, HH1), (LH2, HL2, HH2), (LH3, HL3, HH3) = coeffs

    # Save images for each level
    cv2.imwrite("dwt_LL1.jpg", np.clip(LL1, 0, 255).astype(np.uint8))
    cv2.imwrite("dwt_LH1.jpg", np.clip(LH1, 0, 255).astype(np.uint8))
    cv2.imwrite("dwt_HL1.jpg", np.clip(HL1, 0, 255).astype(np.uint8))
    cv2.imwrite("dwt_HH1.jpg", np.clip(HH1, 0, 255).astype(np.uint8))

    return coeffs


# Encrypt Image using 3-Level DWT + SVD + ACM + Chaos
def encrypt_image(img_path):
    # Read Image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {img_path}")
    
    img = img.astype(np.float64)

    # Apply 3-Level DWT and Save Resolution Images
    coeffs = apply_dwt_and_save(img)
    LL, (LH, HL, HH) = coeffs[0], coeffs[1]

    # Apply SVD on LL Sub-band
    U, S, Vt = svd(LL)

    # Generate Chaotic Sequence
    chaotic_seq = logistic_map(len(S))

    # Encrypt Singular Values
    S_encrypted = S * chaotic_seq  # Multiplication instead of addition

    # Apply Arnold Cat Map on U and V matrices
    U_encrypted = arnold_cat_map(U, iterations=5)
    Vt_encrypted = arnold_cat_map(Vt, iterations=5)

    # Reconstruct Encrypted LL
    LL_encrypted = np.dot(U_encrypted, np.dot(np.diag(S_encrypted), Vt_encrypted))

    # Apply Inverse DWT to Get Encrypted Image
    encrypted_img = pywt.idwt2((LL_encrypted, (LH, HL, HH)), 'db4')

    # Save Encrypted Image
    cv2.imwrite("encrypted_image.jpg", np.clip(encrypted_img, 0, 255).astype(np.uint8))

    return U_encrypted, S_encrypted, Vt_encrypted, chaotic_seq, LH, HL, HH


# Decrypt Image
def decrypt_image(U_encrypted, S_encrypted, Vt_encrypted, chaotic_seq, LH, HL, HH):
    # Recover Singular Values
    S_decrypted = S_encrypted / chaotic_seq  # Inverse of multiplication

    # Apply Inverse Arnold Cat Map to Recover U and V
    U_decrypted = inverse_arnold_cat_map(U_encrypted, iterations=5)
    Vt_decrypted = inverse_arnold_cat_map(Vt_encrypted, iterations=5)

    # Reconstruct Decrypted LL Sub-band
    LL_decrypted = np.dot(U_decrypted, np.dot(np.diag(S_decrypted), Vt_decrypted))

    # Apply Inverse DWT
    decrypted_img = pywt.idwt2((LL_decrypted, (LH, HL, HH)), 'db4')
    
    # Ensure valid pixel values before saving
    decrypted_img = np.clip(decrypted_img, 0, 255).astype(np.uint8)

    # Save Decrypted Image
    cv2.imwrite("decrypted_image.jpg", decrypted_img)

    return decrypted_img


# Example Usage
image_path = "input_image.jpg"
U_enc, S_enc, Vt_enc, chaotic_seq, LH, HL, HH = encrypt_image(image_path)

# Decrypt
decrypted_img = decrypt_image(U_enc, S_enc, Vt_enc, chaotic_seq, LH, HL, HH)

# Show Results
cv2.imshow("Decrypted Image", decrypted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
