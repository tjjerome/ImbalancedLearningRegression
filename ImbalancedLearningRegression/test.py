## load libraries
from gn import gn
from ro import ro
from cnn import cnn
from enn import enn
import pandas

# ## load data
# housing = pandas.read_csv(
    
#     ## http://jse.amstat.org/v19n3/decock.pdf
#     "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv"
# )

# ## conduct ro
# housing_ro = ro(
    
#     data = housing, 
#     y = "SalePrice",
#     #replace = True,
#     #under_samp=False
    
# )

# ## load data
# college = pandas.read_csv(
    
#     ## http://jse.amstat.org/v19n3/decock.pdf
#     "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv"
# )

# ## conduct gn
# college_gn = gn(
    
#     data = college, 
#     y = "Grad.Rate",
#     #replace = True,
#     #under_samp=False
     
# )

# ## load data
# diabetic = pandas.read_csv(
    
#     ## http://jse.amstat.org/v19n3/decock.pdf
#     "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/diabetic_data.csv"
# )

# ## conduct smogn
# diabetic_gn = gn(
    
#     data = diabetic, 
#     y = "num_lab_procedures",
#     #replace = True,
#     #under_samp=False
     
# )

# ## load data
# housing = pandas.read_csv(
    
#     ## http://jse.amstat.org/v19n3/decock.pdf
#     "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv"
# )

# ## conduct cnn
# housing_cnn = cnn(
    
#     data = housing, 
#     y = "SalePrice",
#     #replace = True,
#     #under_samp=False
    
# )

# ## load data
# college = pandas.read_csv(
    
#     ## http://jse.amstat.org/v19n3/decock.pdf
#     "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv"
# )

# ## conduct cnn
# college_cnn = cnn(
    
#     data = college, 
#     y = "Grad.Rate",
#     #replace = True,
#     #under_samp=False
# )

## load data
housing = pandas.read_csv(
    
    ## http://jse.amstat.org/v19n3/decock.pdf
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv"
)

## conduct enn
housing_enn = enn(
    
    data = housing, 
    y = "SalePrice",
    #replace = True,
    #under_samp=False
    
)

## load data
college = pandas.read_csv(
    
    ## http://jse.amstat.org/v19n3/decock.pdf
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv"
)

## conduct enn
college_enn = enn(
    
    data = college, 
    y = "Grad.Rate",
    #replace = True,
    #under_samp=False
)

