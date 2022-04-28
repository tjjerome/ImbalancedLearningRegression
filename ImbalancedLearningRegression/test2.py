import pandas
from adasyn import adasyn

## load data
housing = pandas.read_csv(
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv"
)

## conduct ro
housing_adasyn = adasyn(
    data = housing, 
    y = "SalePrice" 
)

housing_adasyn.to_csv("out.csv")