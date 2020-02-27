#
# Created on Oct 26, 2019
#
# @author: Julian Fortune
# @Description:
#

import glob

import numpy as np
import pandas as pd

def createSummary(filePath):
    dataFrame = pd.read_csv(filePath, index_col=0)

    resultFrame = None

    if "windowSize" in dataFrame.columns:
        resultFrame = pd.DataFrame([], columns=dataFrame.columns)
        for windowSize in [1, 5, 10, 15, 30, 60]:
            windowSizeData = dataFrame.loc[dataFrame["windowSize"] == windowSize]

            filtered = windowSizeData.loc[windowSizeData["filtered"] == True]
            unFiltered = windowSizeData.loc[windowSizeData["filtered"] == False]

            filteredMean = filtered.mean()
            filteredMean["participant"] = "all"
            filteredMean["filtered"] = True

            unFilteredMean = unFiltered.mean()
            unFilteredMean["participant"] = "all"
            unFilteredMean["filtered"] = False

            resultFrame = pd.concat([resultFrame,
                                     pd.DataFrame([filteredMean, unFilteredMean],
                                                  columns=dataFrame.columns)])

    else:
        filtered = dataFrame.loc[dataFrame["filtered"] == True]
        unFiltered = dataFrame.loc[dataFrame["filtered"] == False]

        filteredMean = filtered.mean()
        filteredMean["participant"] = "all"
        filteredMean["filtered"] = True

        unFilteredMean = unFiltered.mean()
        unFilteredMean["participant"] = "all"
        unFilteredMean["filtered"] = False

        # print(filteredMean)

        resultFrame = pd.DataFrame([filteredMean, unFilteredMean], columns=dataFrame.columns)

    print(resultFrame)

    resultFrame.to_csv(filePath[:-4]+"-summary.csv")


def main():
    createSummary("./analyses/supervisoryCrossValidationResults-LeaveOut['filledPauses']-50epochs.csv")
    createSummary("./analyses/supervisoryCrossValidationResults-LeaveOut['respirationRate']-50epochs.csv")
    createSummary("./analyses/supervisoryCrossValidationResults-LeaveOut[]-50epochs.csv")

    createSummary("./analyses/realTimeWindowSizeEvaluation-LeaveOut['filledPauses']-50epochs.csv")
    createSummary("./analyses/realTimeWindowSizeEvaluation-LeaveOut['respirationRate']-50epochs.csv")
    createSummary("./analyses/realTimeWindowSizeEvaluation-LeaveOut[]-50epochs.csv")
    createSummary("./analyses/realTimeWindowSizeEvaluation-LeaveOut['filledPauses', 'respirationRate']-50epochs.csv")

if __name__ == "__main__":
    main()
