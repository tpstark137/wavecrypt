import numpy as np
import cv2
import pywt
from scipy.linalg import svd

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

def logistic_map(size, r=3.9, x0=0.5):
    x = np.zeros(size)
    x[0] = x0
    for i in range(1, size):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return (x * 255).astype(np.uint8)

def encrypt_channel(channel):
    coeffs = pywt.dwt2(channel, 'db4')
    LL, (LH, HL, HH) = coeffs
    U, S, Vt = svd(LL)
    chaotic_seq = logistic_map(len(S))
    S_encrypted = S + chaotic_seq
    U_encrypted = arnold_cat_map(U)
    Vt_encrypted = arnold_cat_map(Vt)
    LL_encrypted = np.dot(U_encrypted, np.dot(np.diag(S_encrypted), Vt_encrypted))
    encrypted_channel = pywt.idwt2((LL_encrypted, (LH, HL, HH)), 'db4')
    return encrypted_channel, U_encrypted, S_encrypted, Vt_encrypted, chaotic_seq, LH, HL, HH

def decrypt_channel(U, S, Vt, chaotic_seq, LH, HL, HH):
    S_decrypted = S - chaotic_seq
    U_decrypted = arnold_cat_map(U, iterations=5)
    Vt_decrypted = arnold_cat_map(Vt, iterations=5)
    LL_decrypted = np.dot(U_decrypted, np.dot(np.diag(S_decrypted), Vt_decrypted))
    decrypted_channel = pywt.idwt2((LL_decrypted, (LH, HL, HH)), 'db4')
    return decrypted_channel

def encrypt_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    channels = cv2.split(img)
    encrypted_channels = []
    encryption_data = []
    for channel in channels:
        encrypted_channel, U, S, Vt, chaotic_seq, LH, HL, HH = encrypt_channel(channel.astype(np.float64))
        encrypted_channels.append(encrypted_channel)
        encryption_data.append((U, S, Vt, chaotic_seq, LH, HL, HH))
    encrypted_img = cv2.merge(encrypted_channels)
    return encrypted_img, encryption_data

def decrypt_image(encryption_data):
    decrypted_channels = []
    for U, S, Vt, chaotic_seq, LH, HL, HH in encryption_data:
        decrypted_channel = decrypt_channel(U, S, Vt, chaotic_seq, LH, HL, HH)
        decrypted_channels.append(decrypted_channel)
    decrypted_img = cv2.merge(decrypted_channels)
    return decrypted_img

image_path = "input_image.jpg"
encrypted_img, encryption_data = encrypt_image(image_path)
cv2.imwrite("encrypted_image.jpg", np.clip(encrypted_img, 0, 255).astype(np.uint8))
decrypted_img = decrypt_image(encryption_data)
cv2.imwrite("decrypted_color_image.jpg", np.clip(decrypted_img, 0, 255).astype(np.uint8))
