import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from radioSignal import radioSignal
from analysis import convolution


class pitchEstimator:
    ### smooth filter
    L = 15
    ### condition of f0
    f_threshold = 100

    def __init__(self, signal:radioSignal, onset) -> None:
        self.signal = signal
        self.onset = onset
    @staticmethod
    def to_midi(f):
        return 48+12*np.log2(f/261.63)
    @classmethod
    def smoothFilter(cls):
        l = np.array(range(0, cls.L))
        r = np.array(range(0, cls.L+1))*(-1) + cls.L
        return np.concatenate([l,r])
    @staticmethod
    def find_fundamental(fft_data, fs, f_threshold=100, maxFactor=0.3, checkFreq=False, title='title', **kwargs):
        l = len(fft_data)
        n0 = int(np.ceil(l/2)) # the fft_data is symmetric
        freq = np.array(range(0,n0))/l * fs
        if checkFreq:
            fig = plt.figure()
            plt.plot(freq, fft_data[0:n0])

        # the maximun frequency is fs
        # and the length of the frequency equal to fft_data
        Xmax = max(fft_data)

        # three condition of being the fundamental freq
        for n in range(1,n0):
            if fft_data[n] > fft_data[n+1] and fft_data[n] > fft_data[n-1]:
                if fft_data[n] > Xmax*maxFactor:
                    if freq[n] > f_threshold:
                        if checkFreq:
                            plt.axvline(freq[n], color="red")
                            plt.title(f'[{title}] f0 = {freq[n]:2f} / {np.round(l)}')
                            plt.show()
                        return freq[n]
        return 1
    
    def plotResult(self, title=""):
        pitch = []
        for n in range(len(self.pitch)):
            pitch = pitch + [self.pitch[n]] * (self.onset[n+1]-self.onset[n])
        fig = plt.figure()
        plt.scatter(range(len(pitch)), pitch, s = 0.3)
        plt.title(title)

    def _guessOnset(self, l, m, r, f_threshold=100, test=False, **kwargs) -> tuple:
        slot1 = self.signal.waveData[l:m]
        slot2 = self.signal.waveData[m:r]
        # fft
        fft_data1 = abs(fft(slot1))/self.signal.fs
        fft_data2 = abs(fft(slot2))/self.signal.fs
        # smooth filter
        fft_data1 = convolution(fft_data1, self.smoothFilter())
        fft_data2 = convolution(fft_data2, self.smoothFilter())
        # find the fundametal frequency
        f1 = self.find_fundamental(fft_data=fft_data1, fs=self.signal.fs,f_threshold=10, title=f'probable f1', **kwargs)
        f2 = self.find_fundamental(fft_data=fft_data2, fs=self.signal.fs,f_threshold=10, title=f'probable f2', **kwargs)
        # convert f0 to MIDI number
        m1 = self.to_midi(f1)
        m2 = self.to_midi(f2)
        
        # prompt
        if test:
            print('midi:', m1, m2)

        return (f1>f_threshold and f2> f_threshold and abs(m1-m2)>1.5 and abs(m1-m2)<12, m1, m2)
    def estimate(self, mode="fft",secondOnsets=False, probableOnset=[], test=False,**kwargs):
        """
            estimate the every section pitch of the audio signal
            mode has:
            fft : fourier transform
            acr : autocorrelation 
            (hht)
        """
        # init the pitch list
        self.pitch = list()
        self.newOnset = list()
        # store the length of onset to calculate the temple
        temp = []
        
        if len(probableOnset) and secondOnsets:
            newOnset = []
            secondOnsetsIndex = 0
            if probableOnset[0] < self.onset[0]:
                slot = self.signal.waveData[probableOnset[0]: self.onset[0]]
                fft_data = abs(fft(slot))/self.signal.fs
                fft_data = convolution(fft_data, self.smoothFilter())
                f = self.find_fundamental(fft_data, self.signal.fs, 100, title=f'prob1 ',**kwargs)
                if f > 100:
                    self.pitch.append(self.to_midi(f))
                    secondOnsetsIndex = secondOnsetsIndex + 1
                    newOnset.append(probableOnset[0])
                    self.newOnset.append(probableOnset[0])
                    temp.append(self.onset[0] - probableOnset[0])
                if test:
                    print(f'[slot0]: {self.to_midi(f)}')

            for i in range(1, len(self.onset)):
                # default the second on set is not a onset
                isOnset = False
                # the interval of radio has to fft
                l = self.onset[i-1]
                r = self.onset[i]
                newOnset.append(l)
                # check is there any probable onset point
                while secondOnsetsIndex < len(probableOnset) and probableOnset[secondOnsetsIndex] < r:
                    if test:
                        print(f'[slot{i}]')
                        print(f'secondOnsetsIndex: {secondOnsetsIndex}')
                    # check if the point is in the interval of l and r
                    # is not remve it, else do the guessing
                    if probableOnset[secondOnsetsIndex] > l:
                        isOnset, midi_a, midi_b = self._guessOnset(l, probableOnset[secondOnsetsIndex], r, test=test,**kwargs)
                    # remove the point out of bound [l, r]
                    secondOnsetsIndex = secondOnsetsIndex + 1

                if isOnset:
                    self.pitch.append(midi_a)
                    self.pitch.append(midi_b)
                    temp.append(probableOnset[secondOnsetsIndex-1] - l)
                    temp.append(r - probableOnset[secondOnsetsIndex-1])
                    newOnset.append(probableOnset[secondOnsetsIndex-1])
                    self.newOnset.append(probableOnset[secondOnsetsIndex-1])
                # normal case:
                else:
                    slot = self.signal.waveData[l:r]
                    # fft
                    fft_data = abs(fft(slot))/self.signal.fs
                    # smooth filter
                    fft_data = convolution(fft_data, self.smoothFilter())
                    # find the fundametal frequency
                    f = self.find_fundamental(fft_data, self.signal.fs, title=f'slot {i}',**kwargs)
                    # convert f0 to MIDI number
                    self.pitch.append(self.to_midi(f))
                    # temple
                    temp.append(r-l)
            # combine new and old onset point
            newOnset.append(r)
            self.onset = newOnset
            # generate temple sequence
            self.temple = []
            self.templeRow = []
            b0 = np.mean(temp)
            for b in temp:
                sRow = np.log2(b/b0)
                s = np.round(sRow)
                self.templeRow.append(np.power(2, sRow))
                self.temple.append(np.power(2,s))     

        else:
            self.setTemple()
            # start from 0-1 slot
            for i in range(1, len(self.onset)):
                # the start and end of the slot
                l = self.onset[i-1]
                r = self.onset[i]
                slot = self.signal.waveData[l:r]

                ### estimate the pitch
                if mode == "fft":
                    # fft
                    fft_data = abs(fft(slot))/self.signal.fs
                    # smooth filter
                    fft_data = convolution(fft_data, self.smoothFilter())
                    # find the fundametal frequency
                    f = self.find_fundamental(fft_data, self.signal.fs, **kwargs)
                    # convert f0 to MIDI number
                    self.pitch.append(self.to_midi(f))
                else:
                    raise ValueError("the pitch estimation mode not exsist")
        

    def setTemple(self):
        self.temple = []
        self.templeRow = []
        temp = []

        for a1, a2 in zip(range(len(self.onset)-1),range(1,len(self.onset))):
            temp.append(self.onset[a2]-self.onset[a1])
        # the reasonable formula
        b0 = np.mean(temp)
        ### the data didn't have half beat is a little strange
        for b in temp:
            sRow = np.log2(b/b0)
            s = np.round(sRow)
            self.templeRow.append(np.power(2, sRow))
            self.temple.append(np.power(2,s))        


if __name__ == "__main__":
    from onsetDetector import onsetDetector
    ### bad onset detection
    # song = "好不好"
    # song = "用心良苦"
    # song = "痛哭的人"
    song = "愛情釀的酒"
### others
    # song = "妹妹背著洋娃娃"
    # song = "志明與春嬌" 
    # song = "擁抱"  
    # song = "被動"
    # song = "還是會寂寞"

    file = "20/20lugo_"+ song + ".wav"
    s = radioSignal("D:/program_ding/hummingdata/"+ file)
    
    detector = onsetDetector(s)
    detector.detect_match(frame=80,overlap=40,thresholdHigh=2.9,thresholdLow=1.5,secondOnsets=2)

    est = pitchEstimator(s, detector.onset)
    est.estimate(checkFreq=False, secondOnsets=True, probableOnset=detector.onsetProbable)

### prompt something
    print(f'song: {song}')
    # print(f'The number of onset :{len(detector.onset)}\n{detector.onset}')
    # print(f'The number of pitch :{len(est.pitch)}\n{np.round(est.pitch,3)}')
    # print(f'The number of temple :{len(est.temple)}\n{np.round(est.temple,3)}')
    # print(f'The raw temple :\n{np.round(est.templeRow, 3)}')

    print(f'The number of onset          :{len(detector.onset)}')
    print(f'The number of new onset      :{len(est.onset)}')
    print(f'The number of probable onset :{len(detector.onsetProbable)}')
    
### plot the graph
    # detector.onset = est.onset
    detector.onsetNew = est.newOnset
    detector.showResults(mode='a-1')
    # est.plotResult()
    plt.show()