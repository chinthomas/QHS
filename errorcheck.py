# input a song name and get the detail process of analysis
# ---- about the signal processing -----
# actual song -> link to the radio source
# humming data -> a button to play it
# onset detection -> plot the result
# pitch estimation -> the pitch result figure
# ---- about the matching algorithm -----
# score of each target
# match table

from onsetDetector import*
from pitchEstimator import*
from melodyMatching import *
import json
if __name__ == "__main__":
### parameter
    mode = "test"
    ### onset 
    frame = 100
    overlap = 50
    min_interval = 0.25
    thresholdHigh = 3
    matchFilter= "paper"
    powerAmp = True
    # enable consider probable onset
    secondOnsets = 1
    thresholdLow = 2
    
    ### dynamic programming
    d = 6
    lamda = 1
    # some function
    turn06 = True
    mainMelodyRow=False
    mainMelodyCol=True
  # others (error testing needed)
    findOptimal=False
  # choose song
  # bad onset
    # song = "好不好"
    # song = "用心良苦"
    # song = "愛情釀的酒"
    song = "痛哭的人"
  # others
    # song = "妹妹背著洋娃娃"
    # song = "志明與春嬌" 
    # song = "擁抱"  
    # song = "被動"
    # song = "還是會寂寞"
### open file
    hr1 = hitRater()
    hr2 = hitRater()
    targetFile = "./target/target.json"
    with open(targetFile, 'r', encoding="big5") as jsfile:
        rows = json.load(jsfile)
    signal = radioSignal("D:/program_ding/hummingdata/20/20lugo_"+song +".wav")
    detector = onsetDetector(signal)
    # detector.detect_match()
    detector.detect_match(frame=frame,overlap=overlap, powerAmp=powerAmp,
                          min_interval=min_interval, matchFilter=matchFilter,
                          thresholdHigh=thresholdHigh,thresholdLow=thresholdLow,secondOnsets=secondOnsets)
    est = pitchEstimator(signal, detector.onset)
    est.estimate(secondOnsets=secondOnsets, probableOnset=detector.onsetProbable, test=True)
    
    detector.onsetNew = est.newOnset
    detector.plotResult(mode='a-1')
    # est.plotResult()

    matcher = songMatcher(song=song,estimator=est, targets=rows)
    matcher.match(findOptimal=findOptimal, turn06=turn06,d=d, lamda=lamda, mainMelodyRow=mainMelodyRow,mainMelodyCol=mainMelodyCol)
    hr2.rate(song=song, score=matcher.score)

    matcher.match_raw()
    hr1.rate(song=song, score=matcher.score_row)

    print(song)
    # print("onset *",len(detector.onset))
    # print(detector.onset, '\n')
    print("pitch *",len(est.pitch))
    print(np.round(est.pitch,3), '\n')
    print("temple *",len(est.temple))
    print(est.temple, '\n')
    print("rank_row",hr1.result)
    print("rank",hr2.result)
    print("rank_first",hr2.first)
    plt.show()
