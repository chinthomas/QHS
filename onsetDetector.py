from radioSignal import radioSignal
import numpy as np
import matplotlib.pyplot as plt
from analysis import convolution, findPeaks, find2Peaks

def getEnvelop(wave_data:radioSignal, frame, overlap,  **kwargs)->radioSignal:
        """ remain the max value in a frame """
        wave_env = radioSignal(list(), wave_data.fs/(frame-overlap))
        n = 0
        while n + frame < wave_data.num_frame :
            slot = wave_data[n: n+frame]
            d = max(max(slot),0)
            wave_env.append(d)
            n = n + frame - overlap
        return wave_env

class onsetDetector:
    ### parameters of different method ###

    ### evnelop ###
    # frame
    # overlap
    ### mag alert ###
    # score_threshold
    # bufferlen
    ### surf method ###
    # M = 10 # the number of point to estimate the second-order polynomial is 2*M + 1

    def __init__(self, wave_data) -> None:
        self.signal = wave_data
        self.onsetProbable = []
        self.onsetNew = []

    ### for surf method
    def __slopeMatrix(self, M):
        """ the matrix to calulate slope"""
        A = list()
        for i in range(-M,M+1):
            A.append([1, i, i**2])
        A = np.array(A)
        return A
    
    def __matchFilter(matchFilter='default', **kwargs):
        """
            Get the match filter for envelop signals
        """

        if matchFilter == "special":
            print("special")
            return np.array([2,2,3,3,0,0,-1,-1,-2,-2,-2,-2])
        elif matchFilter=="paper":
            print("paper")
            return np.array([3,3,4,4,-1,-1,-2,-2,-2,-2,-2,-2])
        elif matchFilter=="second":
            return np.array([4,4,3,3,3,0,0,0,-2,-2,-2,-2])
        return np.array([3,3,4,4,1,1,-2,-2,-2.5,-2.5,-3,-3])
    
    def getEnvSlope(self)->np.ndarray:
        index = self.M
        # to make the data length of envelop and its slope
        # the slope initiate with M zeros 
        slope = np.zeros(self.M)
        # collect the freq and the amplitude
        for index in range(self.M,len(self.envelop)):

            if index >= len(self.envelop) - self.M:
                # add 0 at tail
                slope = np.append(slope,[0],0)
            else:
                # get the second-order polymonial
                # s = [a, b, c]
                # and b is the slope
                y = np.transpose(self.envelop[index-self.M:index+self.M+1])
                s1 = np.matmul(np.transpose(self.A), self.A)
                s2 = np.matmul(np.invert(s1), np.transpose(self.A))
                s = np.matmul(s2, y)*(-1)
                slope = np.append(slope,[s[1]],0)
        return slope
    
    def showResults(self, title="", mode='wave'):
        """
            mode:
            'wave'(defualt): only plot the wave with onset point
            'a-1': plot the envolope, score, and wave+onset point in one figure
            'a-3': plot three figure 
        """
        def plotOnset():
            for i in self.onsetProbable:
                plt.axvline(i/self.signal.fs, color = "blue")
            for i in self.onsetNew:
                plt.axvline(i/self.signal.fs, color = "yellow")
            for i in self.onset:
                plt.axvline(i/self.signal.fs, color = "red")
            plt.plot(self.signal.time, self.signal.waveData)
            plt.xlim(self.signal.time[0], self.signal.time[-1])
            plt.xlabel("time(s)")
            return 0
        
        if mode == 'wave':
            fig_onset = plt.figure()
            # plt.title(title+" (onset detection)")
            plotOnset()
            plt.savefig("./figure/onset_" + self.signal.fname.replace(".wav",""))

        elif mode == 'a-1':
            fig_all = plt.figure()
            # plt.title(f'{title}(onset detection)')
            plt.subplot(311)
            self.env.plotTime()
            plt.subplot(312)
            self.score.plotTime()
            plt.axhline(self.thresholdHigh, color = "red")
            plt.axhline(self.thresholdLow, color = "yellow")
            plt.subplot(313)
            plotOnset()
            plt.savefig("./figure/onset_3in1_" + self.signal.fname.replace(".wav",""))

        elif mode == 'a-3':
            fig1 = plt.figure()
            # plt.title(f'{title}(waveform envelop)')
            self.env.plotTime("./figure/envelop_" + self.signal.fname.replace(".wav",""))
            fig2 = plt.figure()
            # plt.title(f'{title}(after convolution)')
            self.score.plotTime("./figure/score_"+ self.signal.fname.replace(".wav",""))
            self.showResults('wave')

    def detect_surf(self, frame, overlap, M, thresholdHigh, min_interval):
        """ 
            get the onset point 
            return value is the 
        """
        self.thresholdHigh = thresholdHigh
        # get the envelop function
        self.envelop = getEnvelop(self.signal, frame=frame, overlap=overlap)
        # matric for calculate the slope of envelop function
        self.M = M # getEnvSlope will use
        self.A = self.__slopeMatrix(M)
        # get the slope
        self.score = self.getEnvSlope()
        # get the inset point
        self.onset = np.array(findPeaks(self.score, thresholdHigh, wide=min_interval*self.signal.fs))*(frame-overlap)

    def detect_match(self, frame=60, overlap=30, min_interval=0.25, powerAmp=True,
    thresholdHigh=2, secondOnsets=False,thresholdLow=0, **kwargs):
        """ 
            use proposed method to analyze
        """ 
        self.thresholdHigh=thresholdHigh
        self.thresholdLow=thresholdLow

        #1-1 get envelop fuction
        self.env = getEnvelop(self.signal, frame=frame, overlap=overlap, **kwargs)
        #1-2 power amp
        if powerAmp:
            self.env.waveData = self.env.waveData ** 0.6 
        else:
            for i in range(self.env.waveData.size):
                self.env.waveData[i] = max(np.log10(self.env.waveData[i])+1, 0)     
        #2-1 get match filter
        match = onsetDetector.__matchFilter(**kwargs)
        #2-2 convolution
        self.score = radioSignal(convolution(self.env.waveData, match), self.env.fs)
        #3 determine the onset
        if secondOnsets==1:
            self.onset, self.onsetProbable = find2Peaks(
                    signal=self.score.waveData,
                    thresholdHigh=thresholdHigh,
                    thresholdLow=thresholdLow, 
                    wide=int(min_interval*self.signal.fs/2/(frame-overlap)),
                    **kwargs
                )
            self.onset = np.array(self.onset)*(frame-overlap)
            self.onsetProbable = np.array(self.onsetProbable)*(frame-overlap)
        elif secondOnsets == 2:
            self.onset = np.array(
                findPeaks(
                    self.score.waveData, 
                    thresholdHigh, 
                    wide=int(min_interval*self.signal.fs/2/(frame-overlap))
                    ))*(frame-overlap)
            
            # different filter
            match = onsetDetector.__matchFilter(matchFilter="second")
            #2-2 convolution
            self.score = radioSignal(convolution(self.env.waveData, match), self.env.fs)
            self.onsetProbable = np.array(
                findPeaks(
                    self.score.waveData, 
                    thresholdHigh, 
                    wide=int(min_interval*self.signal.fs/2/(frame-overlap))
                    ))*(frame-overlap)
        else:
            self.onset = np.array(
                findPeaks(
                    self.score.waveData, 
                    thresholdHigh, 
                    wide=int(min_interval*self.signal.fs/2/(frame-overlap))
                    ))*(frame-overlap)


if __name__ == "__main__":
### bad onset detection
    # song = "好不好"
    # song = "用心良苦"
    # song = "痛哭的人"
    # song = "愛情釀的酒"
### others
    # song = "妹妹背著洋娃娃"
    song = "志明與春嬌" 
    # song = "擁抱"  
    # song = "被動"
    # song = "還是會寂寞"
    file = "20/20lugo_"+song+".wav"
    signal = radioSignal("D:/program_ding/hummingdata/"+ file)
    detector = onsetDetector(signal)
    
    # detector.detect_match(
    #     frame=100,
    #     overlap=50,
    #     thresholdHigh=2.9,
    #     matchFilter="second", 
    #     # secondOnsets=True,
    #     # thresholdLow = 1.5
    #     )
    # detector.showResults(mode='a-1',title=song)
    # plt.show()
    # detector.onsetProbable = detector.onset
    detector.detect_match(
        frame=100,
        overlap=50,
        thresholdHigh=2.9,
        matchFilter="default", 
        secondOnsets=0,
        # thresholdLow = 1.5
        )
    detector.showResults(mode='a-1',title=song)
    plt.show()
    print(len(detector.onset))
    print("onset: ", detector.onset)
    print(len(detector.onsetProbable))
    print("probable: ", detector.onsetProbable)
    # signal = radioSignal("D:/program_ding/hummingdata/10/lugo_train.wav")
    # detector = onsetDetector(signal)
    # detector.detect_surf(60,30,10,1000,100)
    # detector.showResults(mode='a-1')
    # plt.show()