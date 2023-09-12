import numpy as np
import librosa 

''' 
input: 
    audio: np array of shape of (n, 1)
    noise_factor: float sigma value in [0.001, 0.015] 
output:
    augmented audio 
''' 
def noise_injection(audio, noise_factor):
    noise = np.random.randn(len(audio))
    augmented_audio = np.array([[audio[i][0] + noise[i] * noise_factor] for i in range(0, len(noise))])
    return augmented_audio

''' 
input: 
    audio: np array of shape of (n, 1)
    shift_max: maximum value of shift length idealy it should be less than len(audio) / 4
    shift_direction: a string indicating the shifting direction it should be 'right', 'left', or 'both'
output:
    augmented audio 
''' 
def time_shift(audio, shift_max, shift_direction):
    shift = np.random.randint(shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'rand':
        shift_direction = np.random.randint(0, 2)
    if shift_direction == 1:
        shift = -shift
    augmented_audio = np.roll(audio, shift)
     
    if shift > 0:
        augmented_audio[:shift] = 0 
    else:
        augmented_audio[shift:] = 0 
    return augmented_audio 

''' 
input: 
    audio: np array of shape of (n, 1)
    sr: sampling rate of the audio 
    pitch_factor: int n_steps value in [-4, 4] 
output:
    augmented audio 
''' 
def pitch_shift(audio, sr, pitch_factor):
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_factor)

''' 
input: 
    audio: np array of shape of (n, 1)
    stretch_factor: float gamma value in [0.8, 1.25] 
        if stretch_factor > 1, then the audio is sped up
        if stretch_factor < 1, then the audio is slowed down 
output:
    augmented audio 
''' 
def time_stretch(audio, stretch_factor):
    return librosa.effects.time_stretch(y=audio, rate=stretch_factor)
 
''' 
input: 
    audioi: (np array of shape of (n, 1), label)
    audioj: (np array of shape of (n, 1), label) 
    mixed_factor: float lambda value in [0, 1] 
        if mixed_factor = 0.5, then this function performs sample pairing
output:
    (augmented audio, augmented soft label)
''' 
def mixup(audioi, audioj, mixed_factor):
    xi, yi = audioi 
    xj, yj = audioj 
    x_tilde = mixed_factor * xi + (1 - mixed_factor) * xj 
    y_tilde = mixed_factor * yi + (1 - mixed_factor) * yj 
    return x_tilde, y_tilde 

''' 
input: 
    spec: a np array of shape (tlen, flen) storing the mel spectrogram of the audio
    num_mask: number of masks used in the function
    freq_masking_maxp: the max percentage of masking chosen for frequency dimension
    time_masking_maxp: the max percentaeg of masking chosen for time dimension 
output: 
    a np array of shape (tlen, flen) storing the mel spectrogram of the augmented audio 
''' 
def spec_augment(spec, num_mask=2, freq_masking_maxp=0.15, time_masking_maxp=0.3):
    for i in range(num_mask):
        tlen, flen = np.shape(spec)
        freq_p = random.uniform(0.0, freq_masking_maxp)
        time_p = random.uniform(0.0, time_masking_maxp)
        
        num_freqs_to_mask = int(freq_p * flen)
        num_frames_to_mask = int(time_p * tlen)
         
        fstart = int(np.random.uniform(low=0.0, high=flen - num_freqs_to_mask))
        tstart = int(np.random.uniform(low=0.0, high=tlen - num_frames_to_mask))
         
        spec[:, fstart:fstart + num_freqs_to_mask] = 0
        spec[tstart:tstart + num_frames_to_mask, :] = 0
    
    return spec
    