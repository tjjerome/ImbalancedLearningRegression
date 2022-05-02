import pandas
from adasyn import adasyn
from tomeklinks import tomeklinks
from smote import smote
from smogn import smoter

## load data
housing = pandas.read_csv(
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv"
)

college = pandas.read_csv(
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv"
)


## insurance
insurance = pandas.read_csv(
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/insurance.csv"
)

## conduct adasyn
housing_adasyn = adasyn(
    data = housing, 
    y = "SalePrice" 
)

print(housing_adasyn)

# college_adasyn = adasyn(
#     data = college, 
#     y = "Grad.Rate" 
# )

# college_adasyn2 = adasyn(
#     data = college, 
#     y = "Grad.Rate" 
# )

# housing_adasyn.to_csv("out.csv")

# ## conduct tomeklinks
# housing_tomeklinks = tomeklinks(
#     data = housing, 
#     y = "SalePrice" 
# )

# print(housing)
# print(housing_tomeklinks)

# ## insurance
# insurance = pandas.read_csv(
#     "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/insurance.csv"
# )

# insurance_k = smote(
#     data = insurance,
#     y = "charges",
#     k = 2
# )

# print(insurance)
# print(insurance_k)

# ## red wine
# red_wine = pandas.read_csv(
#     "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/red_wine.csv"
# )

# red_wine_basic = tomeklinks(
#     data = red_wine,
#     y = "quality"
# )

# housing_smote = smote(
#     data = housing, 
#     y = "SalePrice" 
# )

# insurance_k = smote(
#     data = insurance,
#     y = "charges",
#     k = 2
# )
