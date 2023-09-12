
import scipy 
from scipy.fft import fft, ifft, fftfreq
from matplotlib import pyplot as plt 
 
''' 
input:
    sample: a 2d array of shape (n, 1) which describe a sound sample
    sr: sampling rate which is how many measrements are taken in one second 
output:
    3 graphs analysing the audio sample 
''' 
def examine(sample, sr):
    y = list(map(lambda x : float(x[0]), sample))
     
    n = len(y) 
     
    yf = 2 * np.abs(fft(y)) / n 
    xf = fftfreq(n, 1 / sr) 
     
    freq, time, mel_spec = scipy.signal.stft(y, sr) 
     
    plt.figure(figsize=(15, 5)) 
     
    plt.subplot(3, 1, 1) 
    plt.xlim(0, len(y)) 
    plt.plot(y) 
    plt.xlabel("time") 
    plt.ylabel("strain rate") 
     
    plt.subplot(3, 1, 2) 
    plt.colorbar() 
    plt.pcolormesh(time, freq, np.abs(mel_spec), vmin=0, vmax=1, shading='gouraud')
    plt.xlabel("time") 
    plt.ylabel("frequency/Hz") 
    plt.ylim(0, sr / 2) 
     
    plt.subplot(3, 1, 3) 
    plt.plot(xf[:n // 2], yf[:n // 2])
    plt.xlabel("frequency/Hz") 
    plt.ylabel("amplitude") 
    plt.xscale("log") 
     
    plt.show()
    return 0
 
''' 
notice this is a funciton very specific to a particular task
in other words, dont use it 
i put it here as a funciton because i want to make the notebook clean 
''' 
def take_a_glance(feature_set, num_to_show): 
    for i in range(0, num_to_show * 4, 4):
        plt.figure(figsize=(15, 5)) 
        plt.subplot(4, 1, 1)
        plt.title("normal") 
        plt.plot(feature_set[i]) 
        plt.subplot(4, 1, 2)
        plt.title("shift pitch") 
        plt.plot(feature_set[i + 1]) 
        plt.subplot(4, 1, 3)
        plt.title("shift time") 
        plt.plot(feature_set[i + 2]) 
        plt.subplot(4, 1, 4)
        plt.title("noise injection") 
        plt.plot(feature_set[i + 3]) 
        plt.show() 

 
''' 
notice this is a funciton very specific to a particular task
in other words, dont use it 
i put it here as a funciton because i want to make the notebook clean 
''' 
def compare3s(feature_set, label_set):
    normal_features = [feature_set[i] for i in range(0, len(feature_set)) if label_set[i] == 0] 
    walk_features = [feature_set[i] for i in range(0, len(feature_set)) if label_set[i] == 1] 
    run_features = [feature_set[i] for i in range(0, len(feature_set)) if label_set[i] == 2] 
    i = np.random.randint(0, len(run_features)) 

    plt.figure(figsize=(15, 5)) 
    plt.subplot(3, 1, 1) 
    plt.title('normal feature') 
    plt.plot(normal_features[i]) 
    plt.subplot(3, 1, 2) 
    plt.title('walk feature') 
    plt.plot(walk_features[i]) 
    plt.subplot(3, 1, 3) 
    plt.title('run feature') 
    plt.plot(run_features[i]) 
    plt.show() 
     
      
''' 
notice this is a funciton very specific to a particular task
in other words, dont use it 
i put it here as a funciton because i want to make the notebook clean 
''' 
def compare3mfcc(feature_set, label_set):
    normal_features = [feature_set[i][0] for i in range(0, len(feature_set)) if label_set[i] == 0] 
    walk_features = [feature_set[i][0] for i in range(0, len(feature_set)) if label_set[i] == 1] 
    run_features = [feature_set[i][0] for i in range(0, len(feature_set)) if label_set[i] == 2] 
    i = np.random.randint(0, len(run_features)) 

    plt.subplot(3, 1, 1) 
    plt.title('normal feature') 
    plt.pcolormesh(normal_features[i], cmap='hot', shading="gourand") 
    plt.colorbar() 
    plt.subplot(3, 1, 2) 
    plt.title('walk feature') 
    plt.pcolormesh(walk_features[i], cmap='hot', shading="gourand") 
    plt.colorbar() 
    plt.subplot(3, 1, 3) 
    plt.title('run feature') 
    plt.pcolormesh(run_features[i], cmap='hot', shading="gourand") 
    plt.colorbar() 
    plt.show() 

