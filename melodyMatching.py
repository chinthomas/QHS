import sys
import numpy as np
from pitchEstimator import pitchEstimator

class DPTable:
    def __init__(self, query, target) -> None:
        # the humming data midi number
        self.query = query
        # the target midi number, that is any song's midi number
        self.target = target
        # 
        self.score = -1
        self.table = np.zeros((len(self.query)+1, len(self.target)+1))

    def distance(self, i, j, d):
        # the index have to minus one due to the first row and column of the table is defined
        D = abs(self.query[i-1] - self.target[j-1])
        # D(i-1, j)
        top = self.table[i-1][j] + d
        # D(i-1, j-1)
        oblique = self.table[i-1][j-1] + D
        # D(i, j-1)
        left = self.table[i][j-1] + d
        return min(top, oblique, left)

    def makeTable(self, d=4):
        self.score = sys.maxsize
        row = len(self.query)+1 
        col = len(self.target)+1
        for i in range(1,row):
            self.table[i][0] = i*d
        for i in range(1,row):
            for j in range(1, col):
                self.table[i][j] = self.distance(i,j,d)
            
            # find the min score
                if i == row-1 and self.score > self.table[i][j]:
                    self.score = self.table[i][j]
                    self.score_pos = (j-row+1,j)

    def findOptimalTarget(self):
        _,pos = self.score_pos
        if len(self.target[pos-len(self.query):pos])==0:
            print('\n',pos)
        return self.target[pos-len(self.query):pos]
    
    def __str__(self):
        output = " "*6
        for i in range(len(self.target)):
            if self.target[i] >= 0:
                output = output + " " +str(self.target[i]) + " "
            else:
                output = output + str(self.target[i]) + " "
        output = output + "\n"

        for i in range(len(self.query)+1):
            ### first column
            #
            if i >= 1:
                if self.query[i-1] >= 0:
                    output = output + " " +str(self.query[i-1]) + "|"
                else:
                    output = output + str(self.query[i-1]) + "|"
            # the first row no comparison
            else:
                output = output +  "  |"
            
            
            for j in range(len(self.target)+1):
                if self.table[i][j] > 9:
                    output = output + str(int(self.table[i][j])) + " "
                else:
                    output = output + " " + str(int(self.table[i][j])) + " "
            output = output + '\n'
        return  output

class DP_temple(DPTable):
    def __init__(self, query, target, query_beat, target_beat) -> None:
        super().__init__(query, target)
        self.beat_query = query_beat
        self.beat_target = target_beat
    def distance(self, i, j, d):
        beat_error = np.log2(self.beat_query[i-1]/self.beat_target[j-1])
        return super().distance(i, j, d) + abs(beat_error)

class DP_temple2(DPTable):
    def __init__(self, query, target, query_beat, target_beat) -> None:
        super().__init__(query, target)
        self.beat_query = query_beat
        self.beat_target = target_beat
    def distance(self, i, j, d, lamda=0.3):
        beat_error = (self.beat_query[i-1]/self.beat_query[i]/self.beat_target[j-1]*self.beat_target[j])
        return super().distance(i, j, d) + abs(np.log2(beat_error))*lamda

class DP_mainMelody_easy(DPTable):
    def __init__(self, query, target, query_beat, target_beat) -> None:
        super().__init__(query, target)
        self.beat_query = query_beat
        self.beat_target = target_beat
    
    def makeTable(self, d=4, **kwargs):
        # score initially biggest number
        self.score = sys.maxsize
        self.score_pos = (-1, -1)
        # row, col add 1
        row = len(self.query)+1 
        col = len(self.target)+1
        # modified the first row of table
        self.makeFirstRow(d=d, **kwargs)
        # modified the first col of table
        self.makeFirstCol(d=d, **kwargs)
        
        # fill the from table 1*1 to n*m
        for i in range(1,row):
            for j in range(1, col):
                # add penalty to each entry
                self.table[i][j] = self.distance(i,j,d, **kwargs)
                # find the min score
                if i == row-1 and self.score > self.table[i][j] and j > len(self.query)-1:
                    # the minimum score update 
                    self.score = self.table[i][j]
                    # record the index j (here is xth row)
                    self.score_pos = (j-row+1,j)
    
    def distance(self, i, j, d, lamda=1, **kwargs):
        beat_error = (self.beat_query[i-1]/self.beat_query[i]/self.beat_target[j-1]*self.beat_target[j])
        return super().distance(i, j, d) + abs(np.log2(beat_error))*lamda
    
    def makeFirstCol(self, d, mainMelodyCol=False, **kwargs):
        if mainMelodyCol:
            for i in range(1,len(self.query)+1):
                # probable main start of lyrics
                # if self.beat_query[i-1] < 3:
                if self.beat_query[i] < 3:
                    self.table[i][0] = self.table[i-1][0] + d
                else:
                    self.table[i][0] = d
        else:
            for i in range(1,len(self.query)+1):
                self.table[i][0] = i*d
    
    def makeFirstRow(self, d, mainMelodyRow=False, **kwargs):
        if mainMelodyRow:
            for j in range(1,len(self.target)+1):
                # temple >= 2 is d
                if self.beat_target[j-1] >= 2:
                    self.table[0][j] = d
                    self.table[0][j-1] = d
                    self.table[0][j-2] = d
                # temple >= 4 is 2d
                    if self.beat_target[j-1] >= 4:
                        self.table[0][j] = self.table[0][j] + d
                        self.table[0][j-1] = self.table[0][j-1] + d
                        self.table[0][j-2] = self.table[0][j-2] + d  

class hitRater():
    """ 
        record the rank of target song and count the hit rate with the formula hit rate = summation (1/rank)
    """
    def __init__(self) -> None:
        self.hitrate = 0
        # record ranking of target songs after DP
        self.result = dict()

    def rate(self, score:dict, song:str):
        # sort the song by score
        rank = sorted(score.items(), key=lambda item: item[1])
        # record the first place
        self.first = rank[0]
        # find the target song in the all songs
        for k, s in enumerate(rank, 1):
            if s == (song, score[song]):
                self.hitrate = self.hitrate + 1/(k)
                self.result[song] = k

class songMatcher():
    """
    compare a song with all data we have, and gnerate the score of song compare with each target
    INPUT
    -song: the wav file name
    -estimator: contain the temple and pitch
    -targets: contains all data
    OUTPUT
    -score: the score with specified method
    -score_row: the score with pure DP
    """
    def __init__(self, song:str, estimator:pitchEstimator, targets:dict) -> None:
        # humming data songs name
        self.song = song
        # humming data 
        self.query = estimator.pitch
        self.queryPitchInterval = toDifference(estimator.pitch)
        self.beat = estimator.temple
        # target
        self.targets = targets
        # score
        self.score = dict()
        self.score_row = dict()
    def match_raw(self):
        for tname in self.targets.keys():
            targetPitchInterval = toDifference(self.targets[tname]["midi"])
            # targetPitchInterval = toDifferenceRemoveBig(self.targets[tname]["midi"])
            m = DPTable(
                self.queryPitchInterval, 
                targetPitchInterval
            )
            m.makeTable()
            self.score_row[tname] = m.score
        
    def match(self,findOptimal=False, turn06=False, **kwargs):
        # run all songs
        if findOptimal:
            fname = "./DP_result/DP_result_" + self.song +".txt"
            with open(fname, 'a') as f:
                f.writelines(f'\nquery: {self.song}\n')
                f.writelines(f'pitch: {self.query}\n')
                f.writelines(f'interval: {self.queryPitchInterval}\n')
        for tname in self.targets.keys():
            targetPitchInterval = toDifference(self.targets[tname]["midi"])
            ### make the interval to 0~6
            if turn06: 
                targetPitchInterval = RemoveBig(targetPitchInterval)
                self.queryPitchInterval = RemoveBig(self.queryPitchInterval)
            # rum dynamic programming
            mm = DP_mainMelody_easy(
                query=self.queryPitchInterval,
                target=targetPitchInterval,
                query_beat=self.beat,
                target_beat=self.targets[tname]["temple"])
            mm.makeTable(**kwargs)

            ### write the data to file DP_result_<songName>
            if findOptimal:
                # the scope to get the min score in the dynamic programming
                r,l = mm.score_pos
                # avoid r < 0
                if r < 0: r = 0
                # write to file
                with open(fname, 'a') as f:
                    # name and score
                    f.writelines(f'{tname}\n{mm.score}\n')
                    # target midi number
                    f.writelines(f'{self.targets[tname]["midi"][r:l]}\n')
                    # interval of target
                    f.writelines(f'{targetPitchInterval[r:l]}\n')
                    
            # record the score of each song
            self.score[tname] = mm.score

### get the interval of the query
def toDifference(midi):
    n = len(midi)
    # midi[n] - midi [n-1]
    arr = np.array(midi[1:n]) - np.array(midi[0:n-1])
    # turn float to int
    arr = arr.astype('int32')
    return arr
### get the interval of the query and make it to 0~6
def RemoveBig(arr):
    for i in range(len(arr)):
        temp = arr[i]
        if temp < 0:
            temp = -temp
        temp = temp%12
        if temp > 6:
            arr[i] = 7 - (temp%6) 
        else: arr[i] = temp        
    return arr
      
if __name__ == "__main__":
    from onsetDetector import onsetDetector
    from radioSignal import radioSignal
    import json
    queryName = "被動"
    targetName = "痛哭的人"
    targetFile = "./db/db.json"
    with open(targetFile, 'r', encoding="big5") as jsfile:
        rows = json.load(jsfile)

    print(f"humming data from : {queryName}")
    print(f"target is from : {targetName}")
    signal = radioSignal("D:/program_ding/hummingdata/20/20lugo_"+ queryName +".wav")
    ### distinguish the onset point
    detector = onsetDetector(signal)
    detector.detect_match()
    ### estimate the f0 and temple from onset point
    est = pitchEstimator(signal, detector.onset)
    est.estimate()
    est.setTemple()
    print(f'length of pitch : {len(est.pitch)}\n')
    ### 
    query = est.pitch
    queryPitchInterval = RemoveBig(toDifference(est.pitch))
    beat = est.temple
    target = rows[targetName]
    targetPitchInterval = RemoveBig(toDifference(target['midi']))
    m = DP_mainMelody_easy(
        query=queryPitchInterval, 
        target=targetPitchInterval,
        target_beat=target['temple'],
        query_beat=est.temple)
    m.makeTable(d=6,mainMelodyRow=True, mainMelodyCol=True)
    print(f'score : {m.score} {m.score_pos}\n')
    with open("./dpTable.txt", 'w') as writefile:
        writefile.write(f'{m}')



