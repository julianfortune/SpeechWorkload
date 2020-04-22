import glob
import sys
import csv
import os
from datetime import date

import numpy as np
import pandas as pd


# Latex for the correlation graph
correlationLatexString = """
\\centering
\\begin{tabu}{cc|ccc}

\\thline
                                &      & \\multicolumn{3}{c}{\\textbf{Additional Feature Included}} \\\\
                                       \\cline{3-5}

\\textbf{Dataset} &  \\textbf{Condition} & \\textbf{None} &  \\textbf{Filler Utterances} &  \\textbf{Respiration Rate} \\\\
\\hline

\\multirow[c]{4}{*}{Unfiltered}  &         OL &  %s &  %s &  %s \\\\
                                &         NL &  %s &  %s &  %s \\\\
                                &         UL &  %s &  %s &  %s \\\\
                                \\cline{2-5}
                                &   \\allCond &  %s &  %s &  %s \\\\

\\hline
\\multirow[c]{4}{*}{Filtered}    &         OL &  %s &  %s &  %s \\\\
                                &         NL &  %s &  %s &  %s \\\\
                                &         UL &  %s &  %s &  %s \\\\
                                \\cline{2-5}
                                &   \\allCond &  %s &  %s &  %s \\\\
\\thline
\\end{tabu}
"""

def createCorrelationTable(results):
    print("Correlation\n============")

    stringsToPlaceInTable = []

    for row in range(8):
        for table in results:
            correlationValue = table.loc[row, "coefficient"]
            significanceValue = table.loc[row, "significance"]

            if significanceValue < 0.0001:
                significanceMarker = "**"
            elif significanceValue < 0.05:
                significanceMarker = "*"
            else:
                significanceMarker = ""
            #                                               Remove any trailing .000
            valueString = "{0:.3f}".format(correlationValue).rstrip('0').rstrip('.')

            stringsToPlaceInTable.append(valueString + significanceMarker)
            print(stringsToPlaceInTable[-1])
        print("-----")

    print("============")

    return correlationLatexString % tuple(stringsToPlaceInTable)


def createRMSETable(results):
    print("RMSE\t% Err.\t(Mean)\n=====================")

    stringsToPlaceInTable = []

    for row in range(8):
        for table in results:
            RMSEValue = table.loc[row, "RMSE"]

            #                                    Remove any trailing .000
            errorString = "{0:.3f}".format(RMSEValue).rstrip('0').rstrip('.')

            stringsToPlaceInTable.append(errorString)
            print(stringsToPlaceInTable[-1])
        print("-----")

    print("=====================")

    return correlationLatexString % tuple(stringsToPlaceInTable)


def createAccuracyTables(generalPath, reversed=False):
    leaveOut = [['respirationRate', 'filledPauses'], ['filledPauses'], ['respirationRate']]

    if reversed:
        leaveOut[0] = ['filledPauses', 'respirationRate']

    results = []

    for featuresLeftOut in leaveOut:
        path = generalPath % str(featuresLeftOut)
        currentResults = pd.read_csv(path, index_col=0)

        # Specify a window size of 5 if there are multiple window sizes
        if 'windowSize' in currentResults.columns:
            currentResults = currentResults.loc[currentResults['windowSize'] == 5]
            currentResults = currentResults.reset_index().drop(columns=['index'])

        # Guarantee correct sort
        currentResults['overallWorkloadState'] = pd.Categorical(currentResults['overallWorkloadState'], ["ol", "nl", "ul", "all"])

        currentResults = currentResults.sort_values(by='overallWorkloadState', kind='mergesort') # Use mergesort to get stable sort
        currentResults = currentResults.sort_values(by='filtered', kind='mergesort') # Use mergesort to get stable sort
        currentResults = currentResults.reset_index().drop(columns=['index'])

        print(currentResults)

        results.append(currentResults)

    # results = tuple(results) # (=, +filledPauses, +respirationRate)

    CorrelationLatex = createCorrelationTable(results)
    print("% Table created " + date.today().strftime("%B %d, %Y"))
    print("% " + generalPath)
    print(CorrelationLatex)

    RMSELatex = createRMSETable(results)
    print("% Table created " + date.today().strftime("%B %d, %Y"))
    print("% " + generalPath)
    print(RMSELatex)

def main():
    # createAccuracyTables("./analyses/realWorldResults-LeaveOut%s-100epochs.csv")

    # createAccuracyTables("./analyses/peerHumanRobot-LeaveOut%s-100epochs.csv")
    # createAccuracyTables("./analyses/supervisoryHumanRobot-LeaveOut%s-100epochs.csv") # Doesn't work bc peer

    createAccuracyTables("./analyses/supervisoryCrossValidationResults-LeaveOut%s-50epochs-summary.csv")

    # createAccuracyTables("./analyses/realTimeSanityCheck-LeaveOut%s-100epochs.csv")

    # createAccuracyTables("./analyses/realTimeWindowSizeEvaluation-LeaveOut%s-50epochs-summary.csv", reversed=True)


if __name__ == "__main__":
    main()





