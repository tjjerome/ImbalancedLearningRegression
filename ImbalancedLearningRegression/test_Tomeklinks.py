## load libraries
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import ssl

from ImbalancedLearningRegression.tomeklinks import tomeklinks

ssl._create_default_https_context = ssl._create_unverified_context

college = pd.read_csv(
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv"
)

college_tomek = tomeklinks(

    ## main arguments
    data = college,  ## pandas dataframe
    y = "Grad.Rate",  ## string ('header name')
)
college_tomek.reset_index(inplace=True)

seaborn.kdeplot(college["Grad.Rate"], label="Original")
seaborn.kdeplot(college_tomek["Grad.Rate"], label="Modified")
plt.show()
