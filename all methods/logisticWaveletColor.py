import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Logistic Map Function for Chaotic Sequence Generation
def logistic_map(size, key):
    x = key  # Initial condition (0 < key < 1)
    r = 3.99  # Chaos parameter (3.57 < r < 4 for full chaos)
    sequence = np.zeros(size, dtype=np.float64)
    
    for i in range(size):
        x = r * x * (1 - x)  # Logistic Map equation
        sequence[i] = x
    
    # Normalize to 8-bit range (0-255)
    return np.uint8(sequence * 255)

# Encrypt Image using Wavelet Transform & Chaos
def encrypt_image(image_path, key):
    # Load image and convert to YCrCb color space
    img = cv2.imread(image_path)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    channels = cv2.split(img_ycrcb)  # Split into Y, Cr, Cb channels
    encrypted_channels = []
    
    for channel in channels:
        rows, cols = channel.shape
        
        # Apply 2D Discrete Wavelet Transform (DWT)
        coeffs = pywt.dwt2(channel, 'db4')
        LL, (LH, HL, HH) = coeffs  # Extract sub-bands
        
        # Generate chaotic sequence for encryption
        chaos_seq = logistic_map(LL.size, key).reshape(LL.shape)
        
        # Encrypt the LL band using XOR with chaotic sequence
        LL_encrypted = np.bitwise_xor(LL.astype(np.uint8), chaos_seq)
        
        # Reconstruct encrypted channel using inverse DWT
        encrypted_channel = pywt.idwt2((LL_encrypted, (LH, HL, HH)), 'db4')
        encrypted_channel = np.clip(encrypted_channel, 0, 255).astype(np.uint8)
        
        encrypted_channels.append(encrypted_channel)
    
    # Merge encrypted channels and convert back to BGR
    encrypted_img_ycrcb = cv2.merge(encrypted_channels)
    encrypted_img = cv2.cvtColor(encrypted_img_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return encrypted_img

# Decrypt Image using Wavelet Transform & Chaos
def decrypt_image(encrypted_img, key):
    # Convert encrypted image to YCrCb color space
    img_ycrcb = cv2.cvtColor(encrypted_img, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_ycrcb)  # Split into Y, Cr, Cb channels
    decrypted_channels = []
    
    for channel in channels:
        rows, cols = channel.shape
        
        # Apply DWT to encrypted channel
        coeffs = pywt.dwt2(channel, 'db4')
        LL_encrypted, (LH, HL, HH) = coeffs  # Extract sub-bands
        
        # Generate the same chaotic sequence
        chaos_seq = logistic_map(LL_encrypted.size, key).reshape(LL_encrypted.shape)
        
        # Decrypt LL band using XOR with chaotic sequence
        LL_decrypted = np.bitwise_xor(
            LL_encrypted.astype(np.uint8),
            chaos_seq
        )
        
        # Reconstruct decrypted channel using inverse DWT
        decrypted_channel = pywt.idwt2((LL_decrypted, (LH, HL, HH)), 'db4')
        decrypted_channel = np.clip(decrypted_channel, 0, 255).astype(np.uint8)
        
        decrypted_channels.append(decrypted_channel)
    
    # Merge decrypted channels and convert back to BGR
    decrypted_img_ycrcb = cv2.merge(decrypted_channels)
    decrypted_img = cv2.cvtColor(decrypted_img_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return decrypted_img

# =======================
# Test the Encryption and Decryption
# =======================
image_path = 'test_image.jpeg'  # Replace with your image file
key = 0.618  # Must be between (0,1) for Logistic Map

# Encrypt Image
encrypted_img = encrypt_image(image_path, key)
cv2.imwrite("encrypted_image.jpg", encrypted_img)

# Decrypt Image
decrypted_img = decrypt_image(encrypted_img, key)
cv2.imwrite("decrypted_image.jpg", decrypted_img)

# Display Results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Encrypted Image")
plt.imshow(cv2.cvtColor(encrypted_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Decrypted Image")
plt.imshow(cv2.cvtColor(decrypted_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
