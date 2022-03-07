## load libraries
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import ssl

from random_under import random_under

ssl._create_default_https_context = ssl._create_unverified_context

housing = pd.read_csv(
    ## http://jse.amstat.org/v19n3/decock.pdf
    'https://raw.githubusercontent.com/nickkunz/smogn/master/data/housing.csv'
)

housing_ru = random_under(

    ## main arguments
    data = housing,  ## pandas dataframe
    y = "SalePrice",  ## string ('header name')
)
housing_ru.reset_index(inplace=True)

seaborn.kdeplot(housing["SalePrice"], label="Original")
seaborn.kdeplot(housing_ru["SalePrice"], label="Modified")
plt.show()