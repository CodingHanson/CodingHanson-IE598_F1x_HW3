import pandas as pd
import numpy as np
import scipy.stats as stats
import pylab
import sys
# read data
# list 2-1 Sizing up a new data set
with open("C:/Users/wangh/Desktop/MSFE/machine_learning/HW3/HY_Universe_corporate bond.csv") as df:
#arrange data into list for labels and list of lists for attributes
    xList = []
    labels = []
    next(df)
    for line in df:
    #split on comma
        row = line.strip().split(",")
        xList.append(row)
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])))
print ("\n")

#list 2-2 determining the nature of attributes
#arrange data into list for labels and list of lists for attributes

nrow = len(xList)
ncol = len(xList[1])

type = [0]*3
colCounts = []

for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1

    colCounts.append(type)
    type = [0]*3

sys.stdout.write("Col#" + '\t' + "Number" + '\t' +
                 "Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
                     str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1
print ("\n")

#list 2-3 Summary Statistics for Numeric and Categorical Attributes
#generate summary statistics for column 3 (e.g.)
col = 9
colData = []
for row in xList:
    colData.append(float(row[col]))

colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +
            "Standard Deviation = " + '\t ' + str(colsd) + "\n")

#calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
#run again with 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
#The last column contains categorical variables
# target to the type
#I change the position of the col by insert col29 into col36
col = 36
colData = []
for row in xList:
    colData.append(row[col])
unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)
#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
# there are 5 types in the col_29
catCount = [0]*5
for elt in colData:
    catCount[catDict[elt]] += 1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)

#2-4 Quantile-Quantile Plot
import pylab
import scipy.stats as stats
#LiquidityScore
col = 15
colData = []
for row in xList:
    colData.append(float(row[col]))
stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()
#2-5 Read and Summarize
import pandas as pd
from pandas import DataFrame
df_2 = pd.read_csv("C:/Users/wangh/Desktop/MSFE/machine_learning/HW3/HY_Universe_corporate bond.csv",prefix="W")
print(df_2.head())
print(df_2.tail())
#print summary of data frame
summary = df_2.describe()
print(summary)
#2-6 Parallel Coordinates Graph for real Attribute Visualization
# take the value from
df_3 = df_2.drop(['CUSIP','Ticker','Issue Date','Maturity','1st Call Date','Moodys','S_and_P',
              'Fitch','Bloomberg Composite Rating','Maturity Type','Coupon Type','Industry',
              'Months in JNK','Months in HYG','Months in Both','IN_ETF'],axis=1)
print(df_3.head())
cols = list(df_3)
#cols.insert(20,cols.pop(cols.index('bond_type')))
pcolor = []
import matplotlib.pyplot as plot

for row in range(nrow):
    if df_3.iat[row,20] == 1:
        pcolor = "red"
    elif df_3.iat[row,20] == 2:
        pcolor = "blue"
    elif df_3.iat[row,20] == 3:
        pcolor = "green"
    elif df_3.iat[row, 20] == 4:
        pcolor = "yellow"
    elif df_3.iat[row, 20] == 5:
        pcolor = "purple"
    # plot rows of data as if they were series data
    dataRow = df_3.iloc[row, 0:20]
    dataRow.plot(color=pcolor, alpha=0.5)
plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()

#2-7 Cross Plotting Pairs of Attributes
dataRow2 = df_3.iloc[0:nrow,1]
dataRow3 = df_3.iloc[0:nrow,2]
plot.scatter(dataRow2,dataRow3)
plot.xlabel("2nd Attribute")
plot.ylabel(("3rd Attribute"))
plot.show()
dataRow5 = df_3.iloc[0:nrow,4]
plot.scatter(dataRow2, dataRow5)
plot.xlabel("2nd Attribute")
plot.ylabel(("21st Attribute"))
plot.show()

# 2-8 Correlation between Classification Target and Real Attribute
target = []
for row in range(nrow):
    if df_3.iat[row,20] == 1:
        target.append(1.0)
    elif df_3.iat[row,20] == 2:
        target.append(2.0)
    elif df_3.iat[row,20] == 3:
        target.append(3.0)
    elif df_3.iat[row, 20] == 4:
        target.append(4.0)
    elif df_3.iat[row, 20] == 5:
        target.append(5.0)

#plot 16th
dataRow16 = df_3.iloc[0:nrow,15]
plot.scatter(dataRow16,target)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()
#To improve the visualization, this version dithers the points a little
# and makes them somewhat transparent
from random import uniform
target = []
for row in range(nrow):
    if df_3.iat[row,20] == 1:
        target.append(1.0+ uniform(-0.3, 0.3))
    elif df_3.iat[row,20] == 2:
        target.append(2.0+ uniform(-0.3, 0.3))
    elif df_3.iat[row,20] == 3:
        target.append(3.0+ uniform(-0.3, 0.3))
    elif df_3.iat[row, 20] == 4:
        target.append(4.0+ uniform(-0.3, 0.3))
    elif df_3.iat[row, 20] == 5:
        target.append(5.0+ uniform(-0.3, 0.3))

dataRow16 = df_3.iloc[0:nrow,15]
plot.scatter(dataRow16,target, alpha=0.5, s=120)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()

#2-9 Pearson's Correlation Calculation for Attribute 2 vs Attribute 3 & 2 vs 16
from math import sqrt
mean2 = 0.0; mean3 = 0.0; mean16 = 0.0
numElt = len(dataRow2)
for i in range(numElt):
    mean2 += dataRow2[i]/numElt
    mean3 += dataRow3[i]/numElt
    mean16 += dataRow16[i]/numElt

var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (dataRow2[i] - mean2) * (dataRow2[i] - mean2)/numElt
    var3 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3)/numElt
    var21 += (dataRow16[i] - mean16) * (dataRow16[i] - mean16)/numElt

corr16 = 0.0; corr116 = 0.0
for i in range(numElt):
    corr16 += (dataRow2[i] - mean2) * \
              (dataRow3[i] - mean3) / (sqrt(var2*var3) * numElt)
    corr116 += (dataRow2[i] - mean2) * \
               (dataRow16[i] - mean16) / (sqrt(var2*var21) * numElt)

sys.stdout.write("Correlation between attribute 2 and 3 \n")
print(corr16)
sys.stdout.write(" \n")

sys.stdout.write("Correlation between attribute 2 and 21 \n")
print(corr116)
sys.stdout.write(" \n")

#2-10 Presenting Attribute Correlations Visually
#calculate correlations between real-valued attributes

corMat = DataFrame(df_2.corr())

#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()

print("-------------------------------------------------------------------------")
print("My name is Han Wang")
print("My NetID is: 'hanw8'")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")