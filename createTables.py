import glob
import sys
import csv
import os
from datetime import date

import numpy as np
import pandas as pd


# Latex for the correlation graph
correlationLatexStringPeer = """
\\centering
\\begin{tabu}{cc|l}
\\thline
\\textbf{Dataset}       & \\textbf{Condition}  &  \\textbf{Coefficient} \\\\
\\hline
\\multirow[c]{4}{*}{Unfiltered}  &        High  &        %s \\\\
                                &        Low  &        %s \\\\
                                \\cline{2-3}
                                &   \\bothCond &        %s \\\\
\\hline
\\multirow[c]{4}{*}{Filtered}    &        High  &        %s \\\\
                                &        Low  &        %s \\\\
                                \\cline{2-3}
                                &   \\bothCond &        %s \\\\
\\thline
\\end{tabu}
"""

rmseLatexStringPeer="""
\\centering
\\begin{tabu}{cc|lr}
\\thline
\\textbf{Dataset}               & \\textbf{Condition} &   \\textbf{RMSE} &   \\textbf{Percent Error} \\\\
\\hline
\\multirow[c]{3}{*}{Unfiltered} &                 High & %s & %s\\%% \\\\
                               &                 Low & %s & %s\\%% \\\\
                               \\cline{2-4}
                               &           \\bothCond & %s & %s\\%% \\\\
\\hline
\\multirow[c]{3}{*}{Filtered}   &                 High & %s & %s\\%% \\\\
                               &                 Low & %s & %s\\%% \\\\
                               \\cline{2-4}
                               &           \\bothCond & %s & %s\\%% \\\\
\\thline
\\end{tabu}
"""

# Latex for the descriptive statistics graph
descriptiveLatexStringPeer="""
\\centering
\\begin{tabu}{ccc|llll}
\\thline
\\textbf{Dataset} &  \\textbf{Condition} & \\textbf{Source} &  \\textbf{Mean} &  \\textbf{St. Dev.} &  \\textbf{Median} &  \\textbf{Max.} \\\\
\\hline
\\multirow[c]{6}{*}{Unfiltered} & \\multirow[c]{2}{*}{High}       &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
                               \\cline{2-7}
                               & \\multirow[c]{2}{*}{Low}       &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
                               \\cline{2-7}
                               & \\multirow[c]{2}{*}{\\bothCond} &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
\\hline
\\multirow[c]{6}{*}{Filtered}   & \\multirow[c]{2}{*}{High}       &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
                               \\cline{2-7}
                               & \\multirow[c]{2}{*}{Low}       &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
                               \\cline{2-7}
                               & \\multirow[c]{2}{*}{\\bothCond} &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
\\thline
\\end{tabu}
"""

# Latex for the correlation graph
correlationLatexString = """
\\centering
\\begin{tabu}{cc|l}
\\thline
\\textbf{Dataset}       & \\textbf{Condition}  &  \\textbf{Coefficient} \\\\
\\hline
\\multirow[c]{3}{*}{Unfiltered}  &        OL  &        %s \\\\
                                &        NL  &        %s \\\\
                                &        UL  &        %s \\\\
                                \\cline{2-3}
                                &   \\allCond &        %s \\\\
\\hline
\\multirow[c]{3}{*}{Filtered}    &        OL  &        %s \\\\
                                &        NL  &        %s \\\\
                                &        UL  &        %s \\\\
                                \\cline{2-3}
                                &   \\allCond &        %s \\\\
\\thline
\\end{tabu}
"""

rmseLatexString="""
\\centering
\\begin{tabu}{cc|lr}
\\thline
\\textbf{Dataset}               & \\textbf{Condition} &   \\textbf{RMSE} &   \\textbf{Percent Error} \\\\
\\hline
\\multirow[c]{4}{*}{Unfiltered} &                 OL & %s & %s\\%% \\\\
                               &                 NL & %s & %s\\%% \\\\
                               &                 UL & %s & %s\\%% \\\\
                               \\cline{2-4}
                               &           \\allCond & %s & %s\\%% \\\\
\\hline
\\multirow[c]{4}{*}{Filtered}   &                 OL & %s & %s\\%% \\\\
                               &                 NL & %s & %s\\%% \\\\
                               &                 UL & %s & %s\\%% \\\\
                               \\cline{2-4}
                               &           \\allCond & %s & %s\\%% \\\\
\\thline
\\end{tabu}
"""

# Latex for the descriptive statistics graph
descriptiveLatexString = """
\\centering
\\begin{tabu}{ccc|llll}
\\thline
\\textbf{Dataset} &  \\textbf{Condition} & \\textbf{Source} &  \\textbf{Mean} &  \\textbf{St. Dev.} &  \\textbf{Median} &  \\textbf{Max.} \\\\
\\hline
\\multirow[c]{8}{*}{Unfiltered} & \\multirow[c]{2}{*}{OL}       &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
                               \\cline{2-7}
                               & \\multirow[c]{2}{*}{NL}       &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
                               \\cline{2-7}
                               & \\multirow[c]{2}{*}{UL}       &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
                               \\cline{2-7}
                               & \\multirow[c]{2}{*}{\\allCond} &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
\\hline
\\multirow[c]{8}{*}{Filtered}   & \\multirow[c]{2}{*}{OL}       &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
                               \\cline{2-7}
                               & \\multirow[c]{2}{*}{NL}       &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
                               \\cline{2-7}
                               & \\multirow[c]{2}{*}{UL}       &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
                               \\cline{2-7}
                               & \\multirow[c]{2}{*}{\\allCond} &    Model &  %s &  %s &  %s &  %s \\\\
                               &                              & Algorithm &  %s &  %s &  %s &  %s \\\\
\\thline
\\end{tabu}
"""

def convertToLatex(df):
    return df.to_latex(index=False, float_format="%.3f")

def createCorrelationTable(results, peer=False):
    print("Correlation\n============")

    stringsToPlaceInTable = []

    row_count = 6 if peer else 8

    for row in range(row_count):
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

    return (correlationLatexStringPeer if peer else correlationLatexString) % tuple(stringsToPlaceInTable)


def createRMSETable(results, peer=False):
    print("RMSE\t% Err.\t(Mean)\n=====================")

    stringsToPlaceInTable = []

    row_count = 6 if peer else 8

    for row in range(row_count):
        RMSEValue = results.loc[row, "RMSE"]
        trueValue = results.loc[row, "actualMean"]

        percentError = RMSEValue/trueValue * 100

        #                                    Remove any trailing .000
        errorString = "{0:.3f}".format(RMSEValue).rstrip('0').rstrip('.')

        stringsToPlaceInTable.append(errorString)
        print(stringsToPlaceInTable[-1], end="\t")

        #                                           Remove any trailing .000
        percentString = "{0:.3f}".format(percentError).rstrip('0').rstrip('.')

        stringsToPlaceInTable.append(percentString)
        print(stringsToPlaceInTable[-1], end="\t")

        print("{0:.3f}".format(trueValue).rstrip('0').rstrip('.'))

    print("=====================")

    return (rmseLatexStringPeer if peer else rmseLatexString) % tuple(stringsToPlaceInTable)

def createDescriptiveTable(results, peer=False):
    print("Mean\tSD\tMed\tMax\tP Mean\tP SD\tP Med\tP Max\n=============================================================================")

    stringsToPlaceInTable = []

    row_count = 6 if peer else 8

    columns = ['actualMean', 'actualStDev', 'actualMedian', 'actualMax', 'predMean', 'predStDev', 'predMedian', 'predMax']

    for row in range(row_count):
        for col in columns:
            value = results.loc[row, col]

            #                                               Remove any trailing .000
            valueString = "{0:.3f}".format(value).rstrip('0').rstrip('.')

            stringsToPlaceInTable.append(valueString)
            print(stringsToPlaceInTable[-1], end="\t")

        print()

    print("=============================================================================")

    return (descriptiveLatexStringPeer if peer else descriptiveLatexString) % tuple(stringsToPlaceInTable)


def createTables(file, isPeer=False):
    results = pd.read_csv(file, index_col=0)

    # print(results)

    CorrelationLatex = createCorrelationTable(results, peer=isPeer)
    print("% Table created " + date.today().strftime("%B %d, %Y"))
    print("% " + file)
    print(CorrelationLatex)

    RMSELatex = createRMSETable(results, peer=isPeer)
    print("% Table created " + date.today().strftime("%B %d, %Y"))
    print("% " + file)
    print(RMSELatex)

    StatsLatex = createDescriptiveTable(results, peer=isPeer)
    print("% Table created " + date.today().strftime("%B %d, %Y"))
    print("% " + file)
    print(StatsLatex)



def main():
    # 1. Emulated Real-World Conditions Experiment
    # createTables("./analyses/realWorldResults-LeaveOut['respirationRate', 'filledPauses']-100epochs.csv")

    # 2. Population Generalizability
    # createTables("./analyses/supervisoryCrossValidationResults-LeaveOut['respirationRate', 'filledPauses']-50epochs-summary.csv")

    # 3.a. Human-Robot Teaming Generalizability (train Peer-based)
    # createTables("./analyses/peerHumanRobot-LeaveOut['respirationRate', 'filledPauses']-100epochs.csv")

    # 3.b. Human-Robot Teaming Generalizability (train Supervisory)
    # createTables("./analyses/supervisoryHumanRobot-LeaveOut['respirationRate', 'filledPauses']-100epochs.csv", isPeer=True)

    # 4. Real-time Evaluation Sanity Check
    # createTables("./analyses/realTimeSanityCheck-LeaveOut['respirationRate', 'filledPauses']-100epochs.csv")

if __name__ == "__main__":
    main()