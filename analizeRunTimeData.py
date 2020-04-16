#
# Created on September 23, 2019
#
# @author: Julian Fortune
# @Description:
#

import sys, time, glob, os
import pandas as pd

basePath = "./training/"
directories = ["Real_Time-1_second_window", "Real_Time-5_second_window", "Real_Time-10_second_window", "Real_Time-15_second_window", "Real_Time-30_second_window", "Real_Time-60_second_window"]
runTimeColumns = ["intensity", "pitch", "voiceActivity", "syllables", "filledPauses", "processingFeatures", "total"]

def analizeRunTimeData():
    for directory in directories:
        print(basePath + directory + "/features/")
        runTimeFrame = pd.DataFrame(columns=runTimeColumns)
        for path in sorted(glob.iglob(basePath + directory + "/features/*run_time.csv")):
            participant = os.path.basename(path)[:-13]

            averageRunTimesForParticipant = pd.read_csv(path, index_col= 0)
            runTimes = list(averageRunTimesForParticipant.mean(axis = 0))
            runTimes.append(sum(runTimes))

            participantRunTimes = pd.DataFrame([runTimes], columns=runTimeColumns, index=[participant])
            runTimeFrame = runTimeFrame.append(participantRunTimes)

        runTimeFrame = runTimeFrame.drop(["filledPauses", "processingFeatures", "total"], axis = 1)
        runTimeFrame['total'] = runTimeFrame.sum(axis=1)

        analysisColumns = ["intensity", "pitch", "voiceActivity", "syllables", "total"]


        runTimeFrame = runTimeFrame.append(pd.DataFrame([list(runTimeFrame.mean(axis = 0))], columns=analysisColumns, index=["average"]))
        runTimeFrame = runTimeFrame.append(pd.DataFrame([list(runTimeFrame.std(axis = 0))], columns=analysisColumns, index=["stdev"]))
        print(runTimeFrame.loc[['average', 'stdev']])

        # print(runTimeFrame)

def createDataFrame():
    frame = pd.DataFrame(columns=["windowSize", "intensityMean", "intensityStDev",
                                  "pitchMean", "pitchStDev",
                                  "voiceActivityMean", "voiceActivityStDev",
                                  "syllablesMean", "syllablesStDev",
                                  "totalMean", "totalStDev"])

    for directory, windowSize in zip(directories, [1,5,10,15,30,60]):
        runTimeFrame = pd.DataFrame(columns=runTimeColumns)
        for path in sorted(glob.iglob(basePath + directory + "/features/*run_time.csv")):
            participant = os.path.basename(path)[:-13]

            averageRunTimesForParticipant = pd.read_csv(path, index_col= 0)
            runTimes = list(averageRunTimesForParticipant.mean(axis = 0))
            runTimes.append(sum(runTimes))

            participantRunTimes = pd.DataFrame([runTimes], columns=runTimeColumns, index=[participant])
            runTimeFrame = runTimeFrame.append(participantRunTimes)

        runTimeFrame = runTimeFrame.drop(["filledPauses", "processingFeatures", "total"], axis = 1)
        runTimeFrame['total'] = runTimeFrame.sum(axis=1)

        analysisColumns = ["intensity", "pitch", "voiceActivity", "syllables", "total"]

        means  = list(runTimeFrame.mean(axis= 0))
        stDevs = list(runTimeFrame.std(axis= 0))

        row = [windowSize]

        for i in range(len(analysisColumns)):
            row.append(means[i])
            row.append(stDevs[i])

        # runTimeFrame = runTimeFrame.append(pd.DataFrame([], columns=analysisColumns, index=["average"]))
        # runTimeFrame = runTimeFrame.append(pd.DataFrame([], columns=analysisColumns, index=["stdev"]))

        frame.loc[len(frame)] = row

    return frame


def main():
    createDataFrame()

main()
