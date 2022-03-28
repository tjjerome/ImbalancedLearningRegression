## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm

## load dependencies - internal
from ImbalancedLearningRegression.dist_metrics import euclidean_dist, heom_dist, overlap_dist

## adaptively generating minority data samples according to their distributions
def over_sampling_adasyn(

    ## arguments / inputs
    data,     ## training set
    label,    ## label for each observation in the dataset 
    index,    ## index of input data 
    perc,     ## oversampling percentage
    k         ## num of neighs for over-sampling
    ):

    """
    generates synthetic observations and is the primary function underlying the
    over-sampling technique utilized in the higher main function 'adasyn()', the
    4 step procedure for generating synthetic observations is:
    
    1) pre-processing: temporarily removes features without variation, label 
    encodes nominal / categorical features, and subsets the training set into 
    two data sets by data type: numeric / continuous, and nominal / categorical
    
    2) distances: calculates the cartesian distances between all observations, 
    distance metric automatically determined by data type (euclidean distance 
    for numeric only data, heom distance for both numeric and nominal data, and 
    hamming distance for nominal only data) and determine k nearest neighbors
    
    3) over-sampling: 'adasyn' is used to determine the number of new synthetic 
    observations to be generated for each observation according to the ratio of 
    majority class in its k nearest neighbors.
    
    'adasyn' only applies to numeric / continuous features, 
    for nominal / categorical features, synthetic values are generated at random 
    from sampling observed values found within the same feature
    
    4) post processing: restores original values for label encoded features, 
    reintroduces constant features previously removed, converts any interpolated
    negative values to zero in the case of non-negative features
    
    returns a pandas dataframe containing both new and original observations of 
    the training set which are then returned to the higher main function 'adasyn()'

    ref:

    He, H., Bai, Y., Garcia, E. A., & Li, S. (2008, June).
    ADASYN: Adaptive synthetic sampling approach for imbalanced
    learning. In 2008 IEEE international joint conference on neural
    networks (IEEE world congress on computational intelligence)
    (pp. 1322-1328). IEEE.

    """

    ## store dimensions of data subset
    n = len(data)
    d = len(data.columns)
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
    
    ## find non-negative numeric features
    feat_non_neg = [] 
    num_dtypes = ["int64", "float64"]
    
    for j in range(d):
        if data.iloc[:, j].dtype in num_dtypes and any(data.iloc[:, j] > 0):
            feat_non_neg.append(j)
    
    ## find features without variation (constant features)
    feat_const = data.columns[data.nunique() == 1]
    
    ## temporarily remove constant features
    if len(feat_const) > 0:
        
        ## create copy of orignal data and omit constant features
        data_orig = data.copy()
        data = data.drop(data.columns[feat_const], axis = 1)
        
        ## store list of features with variation
        feat_var = list(data.columns.values)
        
        ## reindex features with variation
        for i in range(d - len(feat_const)):
            data.rename(columns = {
                data.columns[i]: i
                }, inplace = True)
        
        ## store new dimension of feature space
        d = len(data.columns)
    
    ## create copy of data containing variation
    data_var = data.copy()
    
    ## create global feature list by column index
    feat_list = list(data.columns.values)
    
    ## create nominal feature list and
    ## label encode nominal / categorical features
    ## (strictly label encode, not one hot encode) 
    feat_list_nom = []
    nom_dtypes = ["object", "bool", "datetime64"]
    
    for j in range(d):
        if data.dtypes[j] in nom_dtypes:
            feat_list_nom.append(j)
            data.iloc[:, j] = pd.Categorical(pd.factorize(
                data.iloc[:, j])[0])
    
    data = data.apply(pd.to_numeric)
    
    ## create numeric feature list
    feat_list_num = list(set(feat_list) - set(feat_list_nom))
    
    ## calculate ranges for numeric / continuous features
    ## (includes label encoded features)
    feat_ranges = list(np.repeat(1, d))
    
    if len(feat_list_nom) > 0:
        for j in feat_list_num:
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    else:
        for j in range(d):
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    
    ## subset feature ranges to include only numeric features
    ## (excludes label encoded features)
    feat_ranges_num = [feat_ranges[i] for i in feat_list_num]
    
    ## subset data by either numeric / continuous or nominal / categorical
    data_num = data.iloc[:, feat_list_num]
    data_nom = data.iloc[:, feat_list_nom]
    
    ## get number of features for each data type
    feat_count_num = len(feat_list_num)
    feat_count_nom = len(feat_list_nom)
    
    ## calculate distance between observations based on data types
    ## store results over null distance matrix of n x n
    dist_matrix = np.ndarray(shape = (n, n))
    
    for i in tqdm(range(n), ascii = True, desc = "dist_matrix"):
        for j in range(n):
            
            ## utilize euclidean distance given that 
            ## data is all numeric / continuous
            if feat_count_nom == 0:
                dist_matrix[i][j] = euclidean_dist(
                    a = data_num.iloc[i],
                    b = data_num.iloc[j],
                    d = feat_count_num
                )
            
            ## utilize heom distance given that 
            ## data contains both numeric / continuous 
            ## and nominal / categorical
            if feat_count_nom > 0 and feat_count_num > 0:
                dist_matrix[i][j] = heom_dist(
                    
                    ## numeric inputs
                    a_num = data_num.iloc[i],
                    b_num = data_num.iloc[j],
                    d_num = feat_count_num,
                    ranges_num = feat_ranges_num,
                    
                    ## nominal inputs
                    a_nom = data_nom.iloc[i],
                    b_nom = data_nom.iloc[j],
                    d_nom = feat_count_nom
                )
            
            ## utilize hamming distance given that 
            ## data is all nominal / categorical
            if feat_count_num == 0:
                dist_matrix[i][j] = overlap_dist(
                    a = data_nom.iloc[i],
                    b = data_nom.iloc[j],
                    d = feat_count_nom
                )
    
    ## determine indicies of k nearest neighbors
    knn_index = [None] * n
    
    for i in range(n):
        knn_index[i] = np.argsort(dist_matrix[i])[0:k + 1]

    ## find knn_index that belong to the bump
    ## and convert new index list to matrix
    temp = []
    for i in range(len(knn_index)):
        if knn_index[i][0] in index:
            temp.append(knn_index[i])

    knn_matrix = np.array(temp)
    
    
    ## total number of new synthetic observations to generate
    n_synth = int(len(index) * (perc - 1))

    ## find the ratio ri = #majority class in neighbourhood/k for each observation
    ri = 0
    r = []
    for i in range(len(knn_matrix)):
        count_majority = 0
        for j in range(1, k+1):
            if label[knn_matrix[i][j]] == -1:
                count_majority += 1
        ri = count_majority/k
        r.append(ri)

    ## normalize ri's so their sum equals to 1
    Rhat_i = []
    for r_value in r:
        rhat_i = r_value / sum(r)
        Rhat_i.append(rhat_i)
    assert(sum(Rhat_i) > 0.99)

    ## calculate the number of new synthetic observations 
    ## that will be generated for each observation
    Gi = []
    for rhat_i in Rhat_i:
        gi = round(rhat_i * n_synth)
        Gi.append(int(gi))

    ## sort index since knn_matrix is stored in ascending order
    index.sort()

    ## create null matrix to store new synthetic observations
    synth_matrix = np.ndarray(shape = ((sum(Gi)), d))

    for i in tqdm(range(len(index)), ascii = True, desc = "index"):
        if Gi[i] > 0:

            num = sum(Gi[:i])

            for j in range(Gi[i]):
                no_minority = True
                ## check if there is at least 1 neighbor belongs to minority class
                for l in range(1, k+1):
                    if label[knn_matrix[i][l]] == 1:
                        no_minority = False

                if no_minority == True:
                    synth_matrix[num + j, 0:(d - 1)] = data.iloc[index[i], 0:(d - 1)]

                if no_minority == False:
                    neigh = int(np.random.choice(
                        a=tuple(range(1, k + 1)),
                        size=1))
                    ## check if the selected neighbor belongs to minority class
                    ## and if not, reselect until this neighbor belongs to minority class
                    while (label[knn_matrix[i][neigh]] != 1):
                        neigh = int(np.random.choice(
                            a=tuple(range(1, k + 1)),
                            size=1))

                    ## conduct synthetic minority over-sampling
                    ## technique for regression (adasyn)
                    diffs = data.iloc[knn_matrix[i, neigh], 0:(d - 1)] - data.iloc[index[i], 0:(d - 1)]
                    synth_matrix[num + j, 0:(d - 1)] = data.iloc[index[i], 0:(d - 1)] + rd.random() * diffs

                    ## randomly assign nominal / categorical features from
                    ## observed cases and selected neighbors
                    for x in feat_list_nom:
                        synth_matrix[num + j, x] = [data.iloc[knn_matrix[i, neigh], x], 
                        data.iloc[index[i], x]][round(rd.random())]

                    ## generate synthetic y response variable by
                    ## inverse distance weighted
                    for z in feat_list_num:
                        a = abs(data.iloc[index[i], z] - 
                            synth_matrix[num + j, z]) / feat_ranges[z]
                        b = abs(data.iloc[knn_matrix[i, neigh], z] - 
                            synth_matrix[num + j, z]) / feat_ranges[z]

                    if len(feat_list_nom) > 0:
                        a = a + sum(data.iloc[index[i], feat_list_nom] != 
                                    synth_matrix[num + j, feat_list_nom])
                        b = b + sum(data.iloc[knn_matrix[i, neigh], feat_list_nom] != 
                                    synth_matrix[num + j, feat_list_nom])

                    if a == b:
                        synth_matrix[num + j,(d - 1)] = data.iloc[index[i], (d - 1)] + data.iloc[
                                            knn_matrix[i, neigh], (d - 1)]/2
                    else:
                        synth_matrix[num + j,(d - 1)] = (b * data.iloc[index[i], (d - 1)] + 
                                            a * data.iloc[knn_matrix[i, neigh], (d - 1)])/(a + b)


    ## create null matrix to store original observations
    original_matrix = np.ndarray(shape = (len(index), d))

    for i in tqdm(range(len(index)), ascii = True, desc = "ori_index"):
        original_matrix[i, 0:(d - 1)] = data.iloc[index[i], 0:(d - 1)]

    ## concatenate new generated synthetic observations with original observations
    final_matrix = np.concatenate((synth_matrix, original_matrix), axis = 0)

    ## convert final matrix to dataframe
    data_new = pd.DataFrame(final_matrix)
    

    ## synthetic data quality check
    if sum(data_new.isnull().sum()) > 0:
        raise ValueError("oops! synthetic data contains missing values")
    
    ## replace label encoded values with original values
    for j in feat_list_nom:
        code_list = data.iloc[:, j].unique()
        cat_list = data_var.iloc[:, j].unique()
        
        for x in code_list:
            data_new.iloc[:, j] = data_new.iloc[:, j].replace(x, cat_list[x])
    
    ## reintroduce constant features previously removed
    if len(feat_const) > 0:
        data_new.columns = feat_var
        
        for j in range(len(feat_const)):
            data_new.insert(
                loc = int(feat_const[j]),
                column = feat_const[j], 
                value = np.repeat(
                    data_orig.iloc[0, feat_const[j]], 
                    len(final_matrix))
            )
    
    ## convert negative values to zero in non-negative features
    for j in feat_non_neg:
        # data_new.iloc[:, j][data_new.iloc[:, j] < 0] = 0
        data_new.iloc[:, j] = data_new.iloc[:, j].clip(lower = 0)
    
    ## return over-sampling results dataframe
    return data_new
