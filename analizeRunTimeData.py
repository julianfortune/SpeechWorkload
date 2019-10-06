#
# Created on September 23, 2019
#
# @author: Julian Fortune
# @Description:
#

import sys, time, glob, os
import wavio
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

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

        runTimeFrame = runTimeFrame.append(pd.DataFrame([list(runTimeFrame.mean(axis = 0))], columns=runTimeColumns, index=["average"]))
        runTimeFrame = runTimeFrame.append(pd.DataFrame([list(runTimeFrame.std(axis = 0))], columns=runTimeColumns, index=["stdev"]))
        print(runTimeFrame)


def main():
    analizeRunTimeData()

main()
