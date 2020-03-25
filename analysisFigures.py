import numpy, os, glob
import matplotlib.pyplot as plt
import pandas as pd

def createSupervisoryRealWorldCorrelationFigure():
    dataFilePath = "./analyses/realWorldResults-LeaveOut['respirationRate', 'filledPauses']-100epochs.csv"
    data = pd.read_csv(dataFilePath, index_col=0)

    workloadConditions = ["UL", "NL", "OL", "All"]

    graphFilteredUnfilteredComparison(data,
                                      'coefficient', "Correlation Coefficient",
                                      workloadConditions, "Workload State",
                                      "lightcoral",
                                      "Pearson Coefficient",
                                      "center right")

def createSupervisoryRealWorldRMSEFigure():
    dataFilePath = "./analyses/realWorldResults-LeaveOut['respirationRate', 'filledPauses']-100epochs.csv"
    data = pd.read_csv(dataFilePath, index_col=0)

    workloadConditions = ["UL", "NL", "OL", "All"]

    graphFilteredUnfilteredComparison(data,
                                      'RMSE', "RMSE",
                                      workloadConditions, "Workload State",
                                      "lightseagreen",
                                      "RMSE",
                                      "upper left")

def createSupervisoryRealWorldDescriptiveStatsFigure():
    dataFilePath = "./analyses/realWorldResults-LeaveOut['respirationRate', 'filledPauses']-100epochs.csv"
    data = pd.read_csv(dataFilePath, index_col=0)

    workloadConditions = ["UL", "NL", "OL", "All"]

    print(data)

    graphPredictedActualComparison(data, workloadConditions)


def graphFilteredUnfilteredComparison(data, value, yAxisLabel, conditions, xAxisLabel, color, graphTitle, legendPosition):
    unfiltered = data[data["filtered"] == False]
    filtered = data[data["filtered"] == True]
    unfilteredValues = []
    filteredValues = []

    for condition in conditions:
        unfilteredValues.append(float(unfiltered[unfiltered['overallWorkloadState'] == condition.lower()][value]))
        filteredValues.append(float(filtered[filtered['overallWorkloadState'] == condition.lower()][value]))

    assert len(conditions) == len(unfilteredValues)
    assert len(conditions) == len(filteredValues)

    x = numpy.linspace(0, len(conditions) - 1, len(conditions))

    width = .3
    spacing = 0.05

    plt.rc('font',**{'family':'serif','serif':['Palatino']})

    figure = plt.figure(figsize=(6,4))

    # Plot the data
    plt.bar(x, unfilteredValues, width, label="Unfiltered", color="white", hatch="///", edgecolor=color)
    plt.bar(x + width + spacing, filteredValues, width, label="Filtered", color=color)

    # Label each pair of bars with the condition
    plt.xticks(x + (width + spacing) / 2, conditions)

    for xValue, yValue in enumerate(unfilteredValues):
        plt.text(xValue, yValue, " " + '{0:.2f}'.format(yValue),
                 color= "black", va= "bottom", ha= "center")

    for xValue, yValue in enumerate(filteredValues):
        plt.text(xValue + (width+spacing), yValue, " " + '{0:.2f}'.format(yValue),
                 color= "black", va= "bottom", ha= "center")

    plt.ylabel(yAxisLabel)
    plt.xlabel(xAxisLabel)
    plt.title(graphTitle)
    plt.margins(0.05, 0.1)

    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=5)
    # plt.subplots_adjust(bottom=0.3)
    plt.legend(loc=legendPosition)
    plt.subplots_adjust(bottom=0.2)

    plt.show()


def graphPredictedActualComparison(data, conditions): # , value, yAxisLabel, conditions, xAxisLabel):
    unfiltered = data[data["filtered"] == False]
    filtered = data[data["filtered"] == True]

    unfilteredActualValues = []
    unfilteredActualStDev = []
    filteredActualValues = []
    filteredActualStDev = []
    unfilteredPredictedValues = []
    unfilteredPredictedStDev = []
    filteredPredictedValues = []
    filteredPredictedStDev = []

    for condition in conditions:
        unfilteredActualValues.append(float(unfiltered[unfiltered['overallWorkloadState'] == condition.lower()]['actualMean']))
        unfilteredActualStDev.append(float(unfiltered[unfiltered['overallWorkloadState'] == condition.lower()]['actualStDev']))

        filteredActualValues.append(float(filtered[filtered['overallWorkloadState'] == condition.lower()]['actualMean']))
        filteredActualStDev.append(float(filtered[filtered['overallWorkloadState'] == condition.lower()]['actualStDev']))

        unfilteredPredictedValues.append(float(unfiltered[unfiltered['overallWorkloadState'] == condition.lower()]['predMean']))
        unfilteredPredictedStDev.append(float(unfiltered[unfiltered['overallWorkloadState'] == condition.lower()]['predStDev']))

        filteredPredictedValues.append(float(filtered[filtered['overallWorkloadState'] == condition.lower()]['predMean']))
        filteredPredictedStDev.append(float(filtered[filtered['overallWorkloadState'] == condition.lower()]['predStDev']))

    # assert len(conditions) == len(unfilteredValues)
    # assert len(conditions) == len(filteredValues)

    x = numpy.linspace(0, len(conditions) - 1, len(conditions))

    width = .15
    spacing = 0.02
    betweenSpacing = 0.04

    plt.rc('font',**{'family':'serif','serif':['Palatino']})

    figure = plt.figure(figsize=(6,4))

    predictedColor="thistle"
    predictedErrorColor="black"
    actualColor="sandybrown"
    actualErrorColor="black"

    hatching = "////"

    errorCapSize = 2

    # Plot the data
    plt.bar(x, unfilteredActualValues, width, yerr=unfilteredActualStDev, capsize=errorCapSize, label="Actual Unfiltered", color="white", hatch=hatching, edgecolor=actualColor, ecolor=actualErrorColor)
    plt.bar(x + width + spacing, unfilteredPredictedValues, width, yerr=unfilteredPredictedStDev, capsize=errorCapSize, label="Predicted Unfiltered", color="white", hatch=hatching, edgecolor=predictedColor, ecolor=predictedErrorColor)
    plt.bar(x + 2 * (width) + betweenSpacing + spacing, filteredActualValues, width, yerr=filteredActualStDev, capsize=errorCapSize, label="Actual Filtered", color=actualColor, ecolor=actualErrorColor)
    plt.bar(x + 3 * (width) + betweenSpacing + 2*spacing, filteredPredictedValues, width, yerr=filteredPredictedStDev, capsize=errorCapSize, label="Predicted Filtered", color=predictedColor, ecolor=predictedErrorColor)

    # Label each pair of bars with the condition
    plt.xticks(x + width/2 + spacing + width + betweenSpacing/2, conditions)

    # for xValue, yValue in enumerate(unfilteredValues):
    #     plt.text(xValue, yValue, " " + '{0:.2f}'.format(yValue),
    #              color= "black", va= "bottom", ha= "center")

    # for xValue, yValue in enumerate(filteredValues):
    #     plt.text(xValue + (width+spacing), yValue, " " + '{0:.2f}'.format(yValue),
    #              color= "black", va= "bottom", ha= "center")

    # plt.ylabel(yAxisLabel)
    # plt.xlabel(xAxisLabel)
    # plt.title(graphTitle)
    plt.margins(0.05, 0.1)

    # # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=5)
    # # plt.subplots_adjust(bottom=0.3)
    # plt.legend(loc=legendPosition)
    plt.legend()
    plt.subplots_adjust(bottom=0.2)

    plt.show()


def main():
    # createSupervisoryRealWorldRMSEFigure()
    # createSupervisoryRealWorldCorrelationFigure()
    createSupervisoryRealWorldDescriptiveStatsFigure()

if __name__ == "__main__":
    main()