import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.fftpack import fft

class radioSignal:
    def __init__(self, *arg) -> None:
        """
            input a wave file,
            input a list of signal with information of frame rate and the number of frame
        """
        if len(arg) == 0:
            self.waveData = np.array(list())
            self.fs = 1
            self.num_frame = 0
        ### init with filepath
        elif len(arg) == 1:
            self.filepath = arg[0]
            self.__load()
            self.fname = self.filepath.split('/')[-1]       
        elif len(arg) == 2:
            self.waveData = arg[0]
            self.fs = arg[1]
            self.num_frame = len(self.waveData)
        elif len(arg) == 3:
            self.waveData = np.array(arg[0])
            self.fs = arg[1]
            self.num_frame = arg[2]
        else:
            raise TypeError("the number of arguments is wrong")
        
        ### get the time axis for plot or find peak
        self.time = self.timeAxis(self.fs, self.num_frame)

    def __load(self) -> None:        
        with wave.open(self.filepath, "rb") as wavefile:
            # get the frame rate
            self.fs = wavefile.getframerate()
            # get the number of frame
            self.num_frame = wavefile.getnframes()
            # get the data, these data are in bytes form
            bytes_data = wavefile.readframes(self.num_frame)
        self.waveData = np.frombuffer(bytes_data, dtype=np.int16)
        # input data with normalization
        self.waveData = self.normalization(self.waveData)

    @staticmethod
    def timeAxis(fs, num_frame):
        return np.arange(0, num_frame) / fs
    
    @staticmethod
    def normalization(data):
        mean = np.mean(data)
        std = np.std(data)
        print(max(data))
        return (data - mean)/(10*std) ### WANT TO MAKE SIGNAL BETWEEN -1 ~ 1
        # return (data - mean)/(max(data) - mean) ### WANT TO MAKE SIGNAL BETWEEN -1 ~ 1

    def plotTime(self, title = ""):
        """plot the signal in time domain"""
        plt.figure()
        plt.plot(self.time, self.waveData)
        plt.xlim(self.time[0], self.num_frame/self.fs)
        if title != "":
            plt.savefig(title)
        plt.show()


    def plotFreq(self, f_thersold=0):
        """plot the signal in frequency domain"""
        fft_data = abs(fft(self.waveData))/self.fs
        # ceiling(the closest and bigger integer), return is float
        n0 = int(np.ceil(self.num_frame/2))
        # centralize the result of DFT
        fft_data = np.concatenate([fft_data[n0:self.num_frame],fft_data[0:n0]]) # concatenate -> joint two array
        ### Frequency ###
        freq=np.concatenate([range(n0-self.num_frame,0),range(0,n0)])*self.fs/self.num_frame
        plt.plot(freq, fft_data)
        if f_thersold: 
            plt.xlim(-f_thersold, f_thersold)
        else:
            plt.xlim(-self.fs/2, self.fs/2)
        plt.savefig("plotFreq")
        plt.show()

    def __getitem__(self, index):
        return self.waveData[index]
    
    def __len__(self):
        return self.num_frame
    
    def __setitem__(self, index, value):
        self.waveData[index] = value
    
    def append(self, data):
        self.waveData = np.append(self.waveData, [data])
        if self.num_frame:
            self.time = np.append(self.time, [self.time[-1] + 1/self.fs])
        else:
            self.time = np.array([0])
        self.num_frame += 1

if __name__ == "__main__":
    b = radioSignal("D:/program_ding/hummingdata/10/lugo_挪威的森林.wav")
    b.plotTime()
    # b = radioSignal("D:/program_ding/hummingdata/10/lugo_朋友.wav")
    # b.plotTime()
    b.plotFreq(1000)