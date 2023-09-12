import scipy 
import librosa 
import numpy as np 
 
 
class Spec:
    def __init__(self, spec, maps):
        self.str2label, self.label2str = maps 
        self.label =  self.str2label[spec[-1]]
        self.duration = spec[:-1] 
    def __str__(self):
        out = "-----spec-----\n"
        out += "duration: " 
        for time in self.duration:
            out += str(time)
            out += " "
        out += "\n" 
        out += "label: " + self.label2str[self.label]
        return out 
         
class Instruction:
    def __init__(self, inst, start_time):
        self.cable_id = int(inst[0])
        self.starting_cumulative_time = time_to_sec(inst[1]) - start_time
        self.label = int(inst[2]) 
        self.duration = int(inst[3]) 
        self.ending_cumulative_time = self.starting_cumulative_time + self.duration

    def __str__(self):
        inst = "-------------------\n" 
        inst += "cable id: " + str(self.cable_id) + "\n"
        inst += "starting cumulative time: " + str(self.starting_cumulative_time) + "\n"
        inst += "ending cumulative time: " + str(self.ending_cumulative_time) + "\n"
        inst += "duration: " + str(self.duration) + "\n"
        inst += "class: " + str(self.label)
        return inst 
     
''' 
input: 
    time: a string which represent time in the form HH:MM:SS, for example: 14:20:30
output:
    the total seconds  
''' 
def time_to_sec(time):
    sections = time.split(":")
    hours = float(sections[0])
    minutes = float(sections[1])
    seconds = float(sections[2])
    return int(3600 * hours + 60 * minutes + seconds)
 
''' 
input: 
    l: a 1d array of samples 
    n: size of a chunk
output:
    a generator of all the chunks 
    use list(divide_chunks(l, n)) to get the desired result 
''' 
def div2chunks(l, n): 
    for i in range(0, len(l), n):
        yield l[i:i + n]

''' 
notice this is a function for a very specific task.
dont count on this funciton
input: 
    raw: a np array of shape (l, 1)
output:
    a np array of the respective mfcc values 
''' 
def raw2mfcc(raw):
    b, a = scipy.signal.butter(8, 0.6)
    filtered_raw = scipy.signal.filtfilt(b, a, [m[0] for m in raw]) 
    downsampled_raw = librosa.resample(np.array(filtered_raw), orig_sr=1000, target_sr=800) 
    mfcc = np.swapaxes(librosa.feature.mfcc(y=downsampled_raw, sr=800, n_mfcc=64, hop_length=16), 0, 1) 
    return mfcc 