__version__ = "0.1.0"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
HOP_SAMPLES = 160  # 10 ms at 16 kHz
WIN_SAMPLES = 400  # 25 ms at 16 kHz - STFT analysis window (Hann), zero-padded to N_FFT
N_FFT = 512        # FFT size: nearest power of two >= WIN_SAMPLES. A power-of-two
                   # transform is a fast radix-2 FFT in every deployment runtime
                   # (JS/Swift/Kotlin/C) instead of a slow Bluestein/DFT for N=400.
                   # The 25 ms analysis window is preserved (win_length=400); the
                   # window is just zero-padded to 512 before the FFT (librosa's
                   # standard convention). Mel features come out 40 x 101 either way.
N_MELS = 40
WINDOW_FRAMES = int(WINDOW_SECONDS * SAMPLE_RATE / HOP_SAMPLES)  # ~100 frames
