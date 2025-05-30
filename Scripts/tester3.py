from scipy.special import erfc
import numpy as np

def Q(x):
    """Q-function using erfc"""
    return 0.5 * erfc(x / np.sqrt(2))

def ser_from_snr(M, snr_db):
    """Compute expected SER for square M-QAM given SNR in dB"""
    if M < 4 or (np.log2(M) % 1) != 0:
        raise ValueError("M must be a power of 2 and >= 4 for square QAM.")
    
    snr_linear = 10 ** (snr_db / 10)
    k = np.log2(M)
    x = np.sqrt(3 * snr_linear / (M - 1))
    ser = 2 * (1 - 1 / np.sqrt(M)) * Q(x)
    return ser

# Example usage
M = 64
snr_db = 25

ser = ser_from_snr(M, snr_db)
print(f"Expected SER for {M}-QAM at {snr_db} dB SNR: {100*ser:.2f}%")
