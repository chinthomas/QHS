from radioSignal import radioSignal
from onsetDetector import onsetDetector
from pitchEstimator import pitchEstimator
from melodyMatching import songMatcher, hitRater
import json
if __name__ == "__main__":
    ### parameter
    ### onset 
    frame = 100
    overlap = 50
    min_interval = 0.25
    thresholdHigh = 2.9
    matchFilter= "default"
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
### the target list
    targetFile = "./db/db.json"
    with open(targetFile, 'r', encoding="big5") as jsfile:
        rows = json.load(jsfile)
    songs = [
        "兩隻老虎",
        "分享",
        "分手吧",
        "好不好",
        "妹妹背著洋娃娃",
        "對面的女孩看過來",
        # "征服",
        "志明與春嬌",
        "愛你一萬年",
        "愛情釀的酒",
        "挪威的森林",
        "擁抱",
        "新不了情",
        "朋友",
        "浪人情歌",
        "溫柔",
        "火車快飛",
        "用心良苦",
        "痛哭的人",
        "約定",
        "自由",
        "至少還有你",
        "花心",
        # "被動",
        # "還是會寂寞",
        # "靜止"
        ]
    hr1 = hitRater()
    hr2 = hitRater()
    for song in songs:    
        print(f"humming data from : {song}")

        ### the signal input
        signal = radioSignal("D:/program_ding/hummingdata/15/15lugo_"+ song +".wav")
        
        ### distinguish the onset point
        detector = onsetDetector(signal)
        detector = onsetDetector(signal)
        detector.detect_match(frame=frame,overlap=overlap, matchFilter=matchFilter,
        powerAmp= powerAmp,min_interval=min_interval,
        thresholdHigh=thresholdHigh,secondOnsets=secondOnsets,thresholdLow=thresholdLow)
        # detector.plotResult('a-1')
        # plt.show()
        ### estimate the f0 and temple from onset point
        est = pitchEstimator(signal, detector.onset)
        est.estimate(secondOnsets=secondOnsets, probableOnset=detector.onsetProbable)
        ### 
        matcher = songMatcher(song=song,estimator=est, targets=rows)
        matcher.match(d=d,turn06=turn06, mainMelodyCol=mainMelodyCol, mainMelodyRow=mainMelodyRow)
        hr2.rate(song=song, score=matcher.score)
        matcher.match_raw()
        hr1.rate(song=song, score=matcher.score_row)

    print(f'\nresult of {len(songs)} songs')
    print(f'hitrate2 : {hr2.hitrate/len(songs)}')
    print(f'hitrate1 : {hr1.hitrate/len(songs)}')
    for r in hr2.result:
        print(r, hr2.result[r])
    outfname = input("please give the output file a name:\n")
    if outfname!="0":
        des = input("the description of this execution:")
        with open(outfname, 'x') as outf:
            outf.writelines(f'#des\n#{des}\n')
            outf.writelines(f"#result of {len(songs)} songs\n")
            outf.writelines(f'hitrate:{hr2.hitrate/len(songs)}\n')
            for r in hr2.result:
                outf.writelines(f'{r}:{hr2.result[r]}\n')
