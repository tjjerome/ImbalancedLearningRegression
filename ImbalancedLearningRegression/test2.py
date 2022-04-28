import pandas
from adasyn import adasyn
from tomeklinks import tomeklinks

## load data
housing = pandas.read_csv(
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv"
)

college = pandas.read_csv(
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv"
)

# ## conduct adasyn
# housing_adasyn = adasyn(
#     data = housing, 
#     y = "SalePrice" 
# )

# college_adasyn = adasyn(
#     data = college, 
#     y = "Grad.Rate" 
# )

# college_adasyn2 = adasyn(
#     data = college, 
#     y = "Grad.Rate" 
# )

# housing_adasyn.to_csv("out.csv")

## conduct tomeklinks
housing_tomeklinks = tomeklinks(
    data = housing, 
    y = "SalePrice" 
)

print(housing)
print(housing_tomeklinks)