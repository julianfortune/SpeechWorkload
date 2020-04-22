import glob
import sys
import csv
import os
from datetime import date

import numpy as np
import pandas as pd

import analizeRunTimeData


# Latex for the correlation graph
correlationLatexString = """
\\centering
\\begin{tabu}{cc|llllll}

\\thline
                                &    & \\multicolumn{6}{c}{\\textbf{Window Size (Seconds)}} \\\\
                                     \\cline{3-8}

\\textbf{Dataset} &  \\textbf{Condition} & \\textbf{1s} &  \\textbf{5s} &  \\textbf{10s} &  \\textbf{15s} &  \\textbf{30s} &  \\textbf{60s} \\\\
\\hline

\\multirow[c]{4}{*}{Unfiltered}  &         OL &  %s &  %s &  %s &  %s &  %s &  %s \\\\
                                &         NL &  %s &  %s &  %s &  %s &  %s &  %s \\\\
                                &         UL &  %s &  %s &  %s &  %s &  %s &  %s \\\\
                                \\cline{2-8}
                                &   \\allCond &  %s &  %s &  %s &  %s &  %s &  %s \\\\

\\hline
\\multirow[c]{4}{*}{Filtered}    &         OL &  %s &  %s &  %s &  %s &  %s &  %s \\\\
                                &         NL &  %s &  %s &  %s &  %s &  %s &  %s \\\\
                                &         UL &  %s &  %s &  %s &  %s &  %s &  %s \\\\
                                \\cline{2-8}
                                &   \\allCond &  %s &  %s &  %s &  %s &  %s &  %s \\\\
\\thline
\\end{tabu}
"""

# Latex for the descriptive statistics graph
runTimeLatexString = """
\\centering
\\begin{tabu}{X|XXXXXX}

\\thline
                        & \\multicolumn{6}{c}{\\textbf{Window Size (Seconds)}} \\\\
                        \\cline{2-7}

\\textbf{Feature}        &  \\textbf{1s} &  \\textbf{5s} &  \\textbf{10s} &  \\textbf{15s} &  \\textbf{30s} &  \\textbf{60s} \\\\
\\hline

\\textbf{Intensity}      &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) \\\\
\\hline

\\textbf{Pitch}          &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) \\\\
\\hline

\\textbf{Voice Activity} &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) \\\\
\\hline

\\textbf{Speech-Rate}    &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) \\\\
\\hline

\\textbf{All Features}   &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) &  %s (%s) \\\\
\\hline


\\thline
\\end{tabu}
"""

def convertToLatex(df):
    return df.to_latex(index=False, float_format="%.3f")

def createCorrelationTable(results):
    print("Correlation\n============")

    stringsToPlaceInTable = []

    for row in range(48):
        correlationValue = results.loc[row, "coefficient"]
        significanceValue = results.loc[row, "significance"]

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

    print("============")

    return correlationLatexString % tuple(stringsToPlaceInTable)


def createRMSETable(results):
    print("RMSE\t% Err.\t(Mean)\n=====================")

    stringsToPlaceInTable = []

    for row in range(48):
        RMSEValue = results.loc[row, "RMSE"]

        #                                    Remove any trailing .000
        errorString = "{0:.3f}".format(RMSEValue).rstrip('0').rstrip('.')

        stringsToPlaceInTable.append(errorString)
        print(stringsToPlaceInTable[-1], end="\\\\t")

    print("=====================")

    return correlationLatexString % tuple(stringsToPlaceInTable)


def createAccuracyTables(file):
    results = pd.read_csv(file, index_col=0)

    # Organize table
    # results = results.loc[results['overallWorkloadState'] == 'all']overallWorkloadState
    results['overallWorkloadState'] = pd.Categorical(results['overallWorkloadState'], ["ol", "nl", "ul", "all"])

    results = results.sort_values(by='overallWorkloadState', kind='mergesort') # Use mergesort to get stable sort
    results = results.sort_values(by='filtered', kind='mergesort') # Use mergesort to get stable sort
    results = results.reset_index().drop(columns=['index'])

    print(results)

    CorrelationLatex = createCorrelationTable(results)
    print("% Table created " + date.today().strftime("%B %d, %Y"))
    print("% " + file)
    print(CorrelationLatex)

    RMSELatex = createRMSETable(results)
    print("% Table created " + date.today().strftime("%B %d, %Y"))
    print("% " + file)
    print(RMSELatex)

def createRunTimeTable():
    timings = analizeRunTimeData.createDataFrame()
    print(timings)

    stringsToPlaceInTable = []

    columns = ["intensityMean", "intensityStDev", "pitchMean", "pitchStDev",
               "voiceActivityMean", "voiceActivityStDev", "syllablesMean",
               "syllablesStDev", "totalMean", "totalStDev"]

    for columnIndex in range(len(columns)//2):
        for index in range(len([1,5,10,15,30,60])):
            print(index, columns[columnIndex*2], columns[columnIndex*2 + 1])
            mean  = timings.loc[index, columns[columnIndex*2]]
            meanString  = "{0:.3f}".format(mean).rstrip('0').rstrip('.')
            if not meanString == "0":
                meanString = meanString.lstrip('0')
            if len(meanString) > 4:
                meanString  = "{0:.2f}".format(mean).rstrip('0').rstrip('.').lstrip('0')
            stringsToPlaceInTable.append(meanString)

            stDev = timings.loc[index, columns[columnIndex*2 + 1]]
            stDevString = "{0:.2f}".format(stDev).rstrip('0').rstrip('.')
            if not stDevString == "0":
                stDevString = stDevString.lstrip('0')
            stringsToPlaceInTable.append(stDevString)

            # print(timings.loc[index, 'windowSize'], meanString, stDevString)

    runTimeLatex = runTimeLatexString % tuple(stringsToPlaceInTable)

    print("% Table created " + date.today().strftime("%B %d, %Y"))
    print(runTimeLatex)

def main():
    createAccuracyTables("./analyses/realTimeWindowSizeEvaluation-LeaveOut['filledPauses', 'respirationRate']-50epochs-summary.csv")
    # createRunTimeTable()

if __name__ == "__main__":
    main()





