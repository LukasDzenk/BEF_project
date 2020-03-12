# Created on Wed Feb 19 18:39:56 2020
# creating a scatter plot for different outcomes 

####### Notes:
# # Describing data reminders:
# https://cheatsheets.quantecon.org/stats-cheatsheet.html
# .describe()
# .info()
# .mean()
# groupby('x').describe()
# .value_counts()

# importing modules 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import random as r

# importing data 
#df = pd.read_excel(r'C:\Users\Lukas\Desktop\Kodinimas\scripts\BF20.xlxs') 
df = pd.read_excel(r'C:\Users\Lukas\Desktop\Python\BF20.xlsx', sheet_name=1) 

####### BET VS GAIN FOR LOW AND HIGH FREQUENCY ########################################

# # Setting a figure size for ONE upcoming figure (in inches) (this must be done before creating any figures)
# plt.figure(figsize=(8, 8))

# # var names: Total gain, Total bet, Low freq 
# # lowfreq data  
# df_gain_lowfreq = df.loc[df['Low_freq'] == 1, 'Avg_gain'] 
# df_bet_lowfreq = df.loc[df['Low_freq'] == 1, 'Avg_bet'] 
# # highfreq data 
# df_gain_highfreq = df.loc[df['Low_freq'] == 0, 'Avg_gain'] 
# df_bet_highfreq = df.loc[df['Low_freq'] == 0, 'Avg_bet'] 
 
# # creating scatter plot 
# # lowfreq 
# plt.scatter(df_gain_lowfreq, df_bet_lowfreq, alpha=0.7, color='green') 
# # highfreq 
# plt.scatter(df_gain_highfreq, df_bet_highfreq, alpha=0.7, color='b') 

# # adjusting 
# plt.title('Average gain vs Average bet by different Frequencies') 
# plt.xlabel('Average gain')
# plt.ylabel('Average bet')
# plt.legend(['Low frequency', 'High frequency'], loc=0)

# # Saving the figure (MUST BE DONE BEFORE PLT.SHOW() OTHERWISE IT'LL A BLANK PICTURE)
# plt.savefig('Frequencies.png', dpi=450)
# # plotting 
# plt.show()


# ####### BET VS GAIN FOR MEN AND WOMEN ##################################################

# # Setting a figure size for ONE upcoming figure (in inches) (this must be done before creating any figures)
# plt.figure(figsize=(8, 8))

# # selecting male data
# df_avg_male_gain = df.loc[df['female'] == 0, 'Avg_gain'] 
# df_avg_male_bet = df.loc[df['female'] == 0, 'Avg_bet'] 
# # selecting female data
# df_avg_fem_gain = df.loc[df['female'] == 1, 'Avg_gain'] 
# df_avg_fem_bet = df.loc[df['female'] == 1, 'Avg_bet'] 
 
# # creating scatter plot 
# # male
# plt.scatter(df_avg_male_gain, df_avg_male_bet, alpha=0.7, color='darkblue') # , color='blue'
# # female 
# plt.scatter(df_avg_fem_gain, df_avg_fem_bet, alpha=0.7, color='crimson') # , color='orange'

# # adjusting 
# plt.title('Average gain vs Average bet by gender') 
# plt.xlabel('Average gain')
# plt.ylabel('Average bet')
# plt.legend(['Males', 'Females'], loc=0)

# # Saving the figure
# plt.savefig('MenWomen.png', dpi=450)
# # plotting 
# plt.show() 

# ####### BET VS GAIN FOR DUTCH AND INTERNATIONALS ########################################

# # Setting a figure size for ONE upcoming figure (in inches) (this must be done before creating any figures)
# plt.figure(figsize=(8, 8))

# # selecting dutch data
# df_avg_dutch_gain = df.loc[df['dutch'] == 1, 'Avg_gain'] 
# df_avg_dutch_bet = df.loc[df['dutch'] == 1, 'Avg_bet'] 
# # selecting international data
# df_avg_int_gain = df.loc[df['dutch'] == 0, 'Avg_gain'] 
# df_avg_int_bet = df.loc[df['dutch'] == 0, 'Avg_bet'] 
 
# # creating scatter plot 
# # international 
# plt.scatter(df_avg_int_gain, df_avg_int_bet, alpha=0.7) # , color='orange'
# # dutch
# plt.scatter(df_avg_dutch_gain, df_avg_dutch_bet, alpha=0.7) # , color='blue'


# # adjusting 
# plt.title('Average gain vs Average bet by nationality') 
# plt.xlabel('Average gain')
# plt.ylabel('Average bet')
# plt.legend(['Internationals', 'Dutch'], loc=0)

# # Saving the figure
# plt.savefig('DutchInternationals.png', dpi=450)
# # plotting 
# plt.show() 

# ####### BET VS GAIN FOR EACH YEAR #######################################################

# # Setting a figure size for ONE upcoming figure (in inches) (this must be done before creating any figures)
# plt.figure(figsize=(8, 8))

# # Evaluating if it's worth plotting
# # df['age'].value_counts()
# # Value counts: 1999 - 20 | 1998 - 20 | 1997 - 14

# # selecting 1995
# df_avg_1995_gain = df.loc[df['age'] == 1995, 'Avg_gain'] 
# df_avg_1995_bet = df.loc[df['age'] == 1995, 'Avg_bet'] 
# # # selecting 1996
# df_avg_1996_gain = df.loc[df['age'] == 1996, 'Avg_gain'] 
# df_avg_1996_bet = df.loc[df['age'] == 1996, 'Avg_bet'] 
# # selecting 1997
# df_avg_1997_gain = df.loc[df['age'] == 1997, 'Avg_gain'] 
# df_avg_1997_bet = df.loc[df['age'] == 1997, 'Avg_bet'] 
# # selecting 1998
# df_avg_1998_gain = df.loc[df['age'] == 1998, 'Avg_gain'] 
# df_avg_1998_bet = df.loc[df['age'] == 1998, 'Avg_bet'] 
# # selecting 1999
# df_avg_1999_gain = df.loc[df['age'] == 1999, 'Avg_gain'] 
# df_avg_1999_bet = df.loc[df['age'] == 1999, 'Avg_bet'] 
# # selecting 2000
# df_avg_2000_gain = df.loc[df['age'] == 2000, 'Avg_gain'] 
# df_avg_2000_bet = df.loc[df['age'] == 2000, 'Avg_bet'] 
 
# # creating scatter plot 
# # selecting 1995
# # plt.scatter(df_avg_1995_gain, df_avg_1995_bet, alpha=0.7)
# # selecting 1996
# # plt.scatter(df_avg_1996_gain, df_avg_1996_bet, alpha=0.7)
# # selecting 1997
# plt.scatter(df_avg_1997_gain, df_avg_1997_bet, alpha=0.7)
# # selecting 1998
# plt.scatter(df_avg_1998_gain, df_avg_1998_bet, alpha=0.7)
# # selecting 1999
# plt.scatter(df_avg_1999_gain, df_avg_1999_bet, alpha=0.7)
# # selecting 2000
# # plt.scatter(df_avg_2000_gain, df_avg_2000_bet, alpha=0.7)

# # adjusting 
# plt.title('Average gain vs Average bet by age groups') 
# plt.xlabel('Average gain')
# plt.ylabel('Average bet')
# plt.legend(['1997', '1998', '1999'], loc=0) # Legend for 1997, 1998, 1999 
# # plt.legend(['1995', '1996', '1997', '1998', '1999', '2000'], loc=0) # Legend for all values

# # Saving the figure
# plt.savefig('Years.png', dpi=450)
# # plotting 
# plt.show() 


####### CORRECT GRAPHS ########################################


# Jitter function for dot visibility (when overlapping)
j = 0.025 # value for jitterness, 0.05 is medium
def rand_jitter(arr): # doesn't work if there's 1 value (eg. 0, in the column, it must have at least 2, because max - min always = 0)
    stdev = 0.01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev
def rand_jitter1(arr): # Version 2 for columns with 1 value (THAT IS NOT 0)
    stdev = j*(max(arr))
    return arr + np.random.randn(len(arr)) * stdev
def rand_jitter0(arr): # Version3 for columns with 1 value (THAT *IS* 0)
    return arr + np.random.randn(len(arr)) * j


####### BET FOR LOW AND HIGH FREQUENCY ########################################
# Setting a figure size for ONE upcoming figure (in inches) (this must be done before creating any figures)
plt.figure(figsize=(3, 6))

# selecting high frequency
df_avg_high_bet = df.loc[df['Low_freq'] == 0, 'Avg_bet'] 
high_f = df.loc[df['Low_freq'] == 0, 'Low_freq']
# selecting low frequency
df_avg_low_bet = df.loc[df['Low_freq'] == 1, 'Avg_bet']
low_f = df.loc[df['Low_freq'] == 1, 'Low_freq']

# # Plotting
# # High frequency
plt.scatter(rand_jitter0(high_f), rand_jitter(df_avg_high_bet), alpha=0.43, color='blue')
# # Low frequency
plt.scatter(rand_jitter1(low_f), rand_jitter(df_avg_low_bet), alpha=0.43, color='green')

# plt.scatter(rand_jitter(df['female']), rand_jitter(df['Avg_bet']), alpha=0.43, color='green')

# adjusting 
plt.title('Average bet by frequency') 
plt.ylabel('Average bet')
plt.xlabel('Frequency')
plt.xticks(ticks=[0, 1], labels=['High', 'Low'])

# # Saving the figure (MUST BE DONE BEFORE PLT.SHOW() OTHERWISE IT'LL A BLANK PICTURE)
# plt.savefig('Frequencies.png', dpi=450)
# # plotting 
# plt.show()


####### BET FOR MALES AND FEMALES ########################################
# Setting a figure size for ONE upcoming figure (in inches) (this must be done before creating any figures)
plt.figure(figsize=(3, 6))

# selecting male data
df_avg_male_bet = df.loc[df['female'] == 0, 'Avg_bet'] 
male = df.loc[df['female'] == 0, 'female']
# selecting female data
df_avg_fem_bet = df.loc[df['female'] == 1, 'Avg_bet']
female = df.loc[df['female'] == 1, 'female']

# # Plotting
# # Males
plt.scatter(rand_jitter0(male), rand_jitter(df_avg_male_bet), alpha=0.43, color='darkblue')
# Females
plt.scatter(rand_jitter1(female), rand_jitter(df_avg_fem_bet), alpha=0.43, color='crimson')

# adjusting 
plt.title('Average bet by gender') 
plt.ylabel('Average bet')
plt.xlabel('Gender')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])

# # Saving the figure (MUST BE DONE BEFORE PLT.SHOW() OTHERWISE IT'LL A BLANK PICTURE)
# plt.savefig('Frequencies.png', dpi=450)
# # plotting 
# plt.show()


####### BET FOR DUTCH AND INTERNATIONALS ########################################
# Setting a figure size for ONE upcoming figure (in inches) (this must be done before creating any figures)
plt.figure(figsize=(3, 6))

# selecting dutch data
df_avg_dutch_bet = df.loc[df['dutch'] == 1, 'Avg_bet'] 
dutch = df.loc[df['dutch'] == 1, 'dutch']
# selecting international data
df_avg_intern_bet = df.loc[df['dutch'] == 0, 'Avg_bet'] 
intern = df.loc[df['dutch'] == 0, 'dutch']
 
# Plotting
# Dutch
plt.scatter(rand_jitter1(dutch), rand_jitter(df_avg_dutch_bet), alpha=0.43, color='darkorange')
# Internationals
plt.scatter(rand_jitter0(intern), rand_jitter(df_avg_intern_bet), alpha=0.43, color='blue')

# adjusting 
plt.title('Average bet by nationality') 
plt.xlabel('Nationality')
plt.ylabel('Average bet')
plt.xticks(ticks=[0, 1], labels=['Internationals', 'Dutch'])

# # Saving the figure
# plt.savefig('DutchInternationals.png', dpi=450)
# # plotting 
# plt.show() 


####### BET FOR DIFFERENT AGES ########################################
# Setting a figure size for ONE upcoming figure (in inches) (this must be done before creating any figures)
plt.figure(figsize=(3, 6))

# # Evaluating if it's worth plotting
# # df['age'].value_counts()
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

# selecting 1999
y1999 =  df.loc[df['age'] == 1999, 'age']
df_avg_1999_bet = df.loc[df['age'] == 1999, 'Avg_bet'] 
# selecting 1998
y1998 = df.loc[df['age'] == 1998, 'age']
df_avg_1998_bet = df.loc[df['age'] == 1998, 'Avg_bet'] 
# selecting 1997
y1997 = df.loc[df['age'] == 1997, 'age']
df_avg_1997_bet = df.loc[df['age'] == 1997, 'Avg_bet'] 
 
# Plotting
# 1999 (21)
plt.scatter(rand_jitter0(y1999), rand_jitter(df_avg_1999_bet), alpha=0.43, color='#1f77b4') # blue color
# 1998 (22)
plt.scatter(rand_jitter0(y1998), rand_jitter(df_avg_1998_bet), alpha=0.43, color='#ff7f0e') # yellow color
# 1997 (23)
plt.scatter(rand_jitter0(y1997), rand_jitter(df_avg_1997_bet), alpha=0.43, color='#2ca02c') # green color

# adjusting 
plt.title('Average bet by age') 
plt.xlabel('Age')
plt.ylabel('Average bet')
plt.xticks(ticks=[1999, 1998, 1997], labels=['21', '22', '23'])

# # Saving the figure
# plt.savefig('DutchInternationals.png', dpi=450)
# # plotting 
# plt.show() 
























