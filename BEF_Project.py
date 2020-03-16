# Created on Sun Feb 23 12:26:54 2020
# BEF Project: Experimental data analysis

# Importing modules
import pandas as pd
# import sklearn.linear_model as lr
import statsmodels.api as sm
# import matplotlib.pyplot as plt
from stargazer.stargazer import Stargazer as sg
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import researchpy as rp # for correlation matrix

# importing data 
#df = pd.read_excel(r'C:\Users\Lukas\Desktop\Kodinimas\scripts\BF20.xlxs') 
df = pd.read_excel(r'C:\Users\Lukas\OneDrive\University\Year 3\Behavioral Finance\Project\Python\BEF_Project\BF20.xlsx', sheet_name=1) 

######### Preparing data

## Inspecting data
# Histogram
df['Avg_bet'].hist(bins=100)

# 1999    20
# 1998    20
# 1997    14
# 1996     6
# 1995     6
# 2000     5
# 1994     3
# 1989     1
# 1988     1
# 1986     1

# Time variable (1-6, each point equals 1 year older)
df['Age'] = 0
df.loc[df['age'] == 1994, 'Age'] = 7
df.loc[df['age'] == 1995, 'Age'] = 6
df.loc[df['age'] == 1996, 'Age'] = 5
df.loc[df['age'] == 1997, 'Age'] = 4
df.loc[df['age'] == 1998, 'Age'] = 3
df.loc[df['age'] == 1999, 'Age'] = 2
df.loc[df['age'] == 2000, 'Age'] = 1
df = df.loc[df['Age'] != 0, :]

# Checking for correlation
corr_type, corr_matrix, corr_ps = rp.corr_case(df[['Low_freq', 'female', 'dutch', 'Age']])
# Long story in short, multicollinearity increases the estimate of standard error of regression coefficients which makes some variables statistically insignificant when they should be significant.
# 0.0 – 0.2	Weak correlation
# 0.3 – 0.6	Moderate correlation
# 0.7 – 1.0	Strong correlation

# df.loc[df['female'] == 1, ['female', 'dutch']].groupby(['dutch']).count() # how many females that are dutch and non dutch
# df.loc[df['female'] == 0, ['female', 'dutch']].groupby(['dutch']).count() # How many males that are dutch and non dutch

# Female and dutch (-0.31 correlation | 0.0055 p-value) = females tend to be non dutch

########################################
######### Regressions ##################
########################################
# Sklearn is more developed, however it doesn't output nice summary of traditional statistics

# Regression variables
X = df[['Low_freq', 'female', 'dutch', 'Age']]
Y = df['Avg_bet']

# Running the regression
X = sm.add_constant(X) ## add an intercept (beta_0) to our model
model = sm.OLS(Y, X).fit(cov_type="HC1") # cov_type="HC1" makes the regression model robust

# Calculating predicted values and assigning them to the df
predictions = model.predict(X)
df['Predicted_values'] = predictions
pred = df['Predicted_values']
# Predicted values vs actual values
# plt.scatter(pred, df['Avg_bet'])

# Print out the statistics
print(model.summary())

######### Outputing as LaTeX (latech proununciation)

# Adding extra information to the file
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

# Writting into a file (REGRESSION SUMMARY TABLE)
f = open('Regression_Summary_Table.tex', 'w')
f.write(beginningtex)
f.write(model.summary().as_latex())
f.write(endtex)
f.close()

# Convert LaTeX file into a .pdf (it needs to be compiled, however it can be done through an online compiler) # This is for a summary table output
# overleaf.com


## Scientific paper statistics output using the stargazer module 

# Multiple regression output at once EXAMPLE ONLY:
# est = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:4]])).fit()
# est2 = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:6]])).fit()
# stargazer = Stargazer([est, est2])
# stargazer.render_html()
# stargazer.render_latex()

# Enable this:
stargazer = sg([model])
sr_render = stargazer.render_latex()

# Writting into a file (SCIENTIFIC OUTPUT)
f = open('Regression_Scientific_output.tex', 'w')
f.write(beginningtex)
f.write(sr_render)
f.write(endtex)
f.close()


########################################
#### NON PARAMETRIC TESTS ##############
########################################

### AVG_BET BY FREQUENCY ###
# (MANN-WHITNEY U : BECAUSE OF NON GAUSSIAN DISTRIBUTION)

# It is comparing the two means. 
# p <= alpha (0.05): reject H0, different distribution.
# p > alpha (0.05): fail to reject H0, same distribution.

# Prepping by separating data into two samples
df_avg_high_bet = df.loc[df['Low_freq'] == 0, 'Avg_bet'] 
df_avg_low_bet = df.loc[df['Low_freq'] == 1, 'Avg_bet']

# Clear previous figures
plt.clf()
# Plotting histograms
plt.hist(df_avg_high_bet, bins=30, alpha=0.5, label='High frequency', color='darkgreen')
plt.hist(df_avg_low_bet, bins=30, alpha=0.5, label='Low frequency', color='orange')
plt.ylabel('Value count')
plt.xlabel('Average bet')
plt.legend(loc='upper right')
# # Saving the figure
plt.savefig('hist_bet_by_freq.png', dpi=450)
# plotting 
plt.show() 

# Testing
u_statistic, p_value = stats.mannwhitneyu(df_avg_high_bet, df_avg_low_bet)
# Printing results
print('\n### Mann-Whitney U test:')
print(f'p-value: {p_value:.5f}')

# p value is 0.001, therefore we reject the null hypothesis (sample distributions are equal), otherwise we would fail to reject Ho that the sample distributions are equal.
# Interpret
alpha = 0.05
if p_value > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
# lower than 0.05 p-value rejects the null hypothesis


### AVG_BET BY GENDER ###
# (MANN-WHITNEY U : BECAUSE OF NON GAUSSIAN DISTRIBUTION)

# It is comparing the two means. 
# p <= alpha (0.05): reject H0, different distribution.
# p > alpha (0.05): fail to reject H0, same distribution.

# Prepping by separating data into two samples
df_avg_male_bet = df.loc[df['female'] == 0, 'Avg_bet'] 
df_avg_female_bet = df.loc[df['female'] == 1, 'Avg_bet']

# Clear previous figures
plt.clf()
# Plotting histograms
plt.hist(df_avg_male_bet, bins=30, alpha=0.5, label='Males', color='darkgreen')
plt.hist(df_avg_female_bet, bins=30, alpha=0.5, label='Females', color='orange')
plt.ylabel('Value count')
plt.xlabel('Average bet')
plt.legend(loc='upper right')
# # Saving the figure
plt.savefig('hist_bet_by_gender.png', dpi=450)
# plotting 
plt.show() 

# Testing
u_statistic, p_value = stats.mannwhitneyu(df_avg_male_bet, df_avg_female_bet)
# Printing results
print('\n### Mann-Whitney U test:')
print(f'p-value: {p_value:.5f}')

# p value is 0.001, therefore we reject the null hypothesis (sample distributions are equal), otherwise we would fail to reject Ho that the sample distributions are equal.
# Interpret
alpha = 0.05
if p_value > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
# lower than 0.05 p-value rejects the null hypothesis


# 1. Save .py files 
# 2. Save figures from BEF_visualization (HD export to .png)
# 3. Save figures from BEF_project
# 3. Put Scien2 .tex files into the online converter
# 5. Export .pdf files (open them in word)
# 6. Copy the non-parametric test results to comb_file
# 7. Copy images to comb_file
# 8. Push to git



























