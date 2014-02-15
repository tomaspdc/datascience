# full example extracted from this walkthrough:
# http://blog.yhathq.com/posts/logistic-regression-and-python.html

import pandas
import statsmodels.api
import pylab
import numpy
from itertools import product

# read the data in
df = pandas.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")

# take a look at the dataset
#print(df.head())

# rename the 'rank' column because there is also a
# DataFrame method called 'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]

# list the columns
#print(df.columns)

# explore count/mean/std/min/25%/50%/75%
#print(df.describe())

# std (you just saw this in the line before but whatever)
#print(df.std())

# cross-counts ranks/prestiges against admit/no-admit
#print(pandas.crosstab(df['admit'], df['prestige'], rownames=['admit']))

# creates histograms for each variable and shows them with pylab
#df.hist()
#pylab.show()

# we dummify rank/prestiges with a sparse matrix:
dummy_ranks = pandas.get_dummies(df['prestige'], prefix='prestige')
#print(dummy_ranks.head())

# cleans the data frame (no dummies)
keep = ['admit', 'gre', 'gpa']
data = df[keep].join(dummy_ranks.ix[:, 'prestige_2':])
# > print(data.head())
#    admit  gre   gpa  prestige_2  prestige_3  prestige_4
# 0      0  380  3.61           0           1           0
# 1      1  660  3.67           0           1           0
# 2      1  800  4.00           0           0           0
# 3      1  640  3.19           0           0           1
# 4      0  520  2.93           0           0           1

# manually add the intercept (don't know why it's needed, this is alpha)
data['intercept'] = 1.0

train_cols = data.columns[1:]  # everything but 'admit', which is column[0]

logit = statsmodels.api.Logit(data['admit'], data[train_cols])

result = logit.fit()

# beautiful summary
print(result.summary())

# look at the confidence interval of each coeffecient
print(result.conf_int())

# odds ratios only (exponential)
print(numpy.exp(result.params))

# odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print(numpy.exp(conf))

# now we go for plots, fixing gpa or gre

# instead of generating all possible values of GRE and GPA, we're going
# to use an evenly spaced range of 10 values from the min to the max
gres = numpy.linspace(data['gre'].min(), data['gre'].max(), 10)
print(gres)

gpas = numpy.linspace(data['gpa'].min(), data['gpa'].max(), 10)
print(gpas)

# enumerate all possibilities
combos = pandas.DataFrame(list(product(gres, gpas, [1, 2, 3, 4], [1.])))
combos.columns = ['gre', 'gpa', 'prestige', 'intercept']
dummy_ranks = pandas.get_dummies(combos['prestige'], prefix='prestige')
dummy_ranks.columns = ['prestige_1', 'prestige_2', 'prestige_3', 'prestige_4']

# keep only what we need for making predictions
cols_to_keep = ['gre', 'gpa', 'prestige', 'intercept']
combos = combos[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])

# make predictions on the enumerated dataset
combos['admit_pred'] = result.predict(combos[train_cols])

print(combos.head())

def isolate_and_plot(variable):
    # isolate gre and class rank
    grouped = pandas.pivot_table(
        combos,
        values=['admit_pred'],
        rows=[variable, 'prestige'],
        aggfunc=numpy.mean
    )
    # in case you're curious as to what this looks like
    # print grouped.head()
    #                      admit_pred
    # gre        prestige
    # 220.000000 1           0.282462
    #            2           0.169987
    #            3           0.096544
    #            4           0.079859
    # 284.444444 1           0.311718

    # make a plot
    colors = 'rbgyrbgy'
    for col in combos.prestige.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1) == col]
        pylab.plot(
            plt_data.index.get_level_values(0),
            plt_data['admit_pred'],
            color=colors[int(col)]
        )

    pylab.xlabel(variable)
    pylab.ylabel("P(admit=1)")
    pylab.legend(['1', '2', '3', '4'], loc='upper left', title='Prestige')
    pylab.title("Prob(admit=1) isolating " + variable + " and presitge")
    pylab.show()

isolate_and_plot('gre')
isolate_and_plot('gpa')
