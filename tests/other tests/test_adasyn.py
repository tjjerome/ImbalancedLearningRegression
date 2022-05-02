## load libraries
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import ssl

from ImbalancedLearningRegression.adasyn import adasyn

ssl._create_default_https_context = ssl._create_unverified_context

college = pd.read_csv(
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv"
)

college_adasyn = adasyn(
    ## main arguments
    data=college,  ## pandas dataframe
    y='Grad.Rate'  ## string ('header name')
)
college_adasyn.reset_index(inplace=True)

seaborn.kdeplot(college['Grad.Rate'], label="Original")
seaborn.kdeplot(college_adasyn['Grad.Rate'], label="Modified")
plt.show()
