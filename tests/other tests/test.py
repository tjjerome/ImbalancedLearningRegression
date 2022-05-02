## load libraries
from gn import gn
from ro import ro
from cnn import cnn
from enn import enn
import pandas

## load data
housing = pandas.read_csv(
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv"
)

## conduct ro
housing_ro = ro(
    data = housing, 
    y = "SalePrice" 
)

## conduct gn
housing_gn = gn(
    data = housing, 
    y = "SalePrice" 
)

## conduct cnn
housing_cnn = cnn(
    data = housing, 
    y = "SalePrice" 
)

## conduct enn
housing_enn = enn(
    data = housing, 
    y = "SalePrice" 
)

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
#     n_seed=3000,
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
#     n_seed=3,
#     #replace = True,
#     #under_samp=False
# )

# ## load data
# housing = pandas.read_csv(
    
#     ## http://jse.amstat.org/v19n3/decock.pdf
#     "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv"
# )

# ## conduct enn
# housing_enn = enn(
    
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

# ## conduct enn
# college_enn = enn(
    
#     data = college, 
#     y = "Grad.Rate",
#     #replace = True,
#     #under_samp=False
# )

import pandas
from ImbalancedLearningRegression.ro import ro
from ImbalancedLearningRegression.gn import gn
from ImbalancedLearningRegression.cnn import cnn
from ImbalancedLearningRegression.enn import enn
