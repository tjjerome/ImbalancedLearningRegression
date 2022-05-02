## load dependencies - third party
import numpy as np
import pandas as pd
from tqdm import tqdm

## load dependencies - internal
from ImbalancedLearningRegression.phi import phi
from ImbalancedLearningRegression.phi_ctrl_pts import phi_ctrl_pts
from ImbalancedLearningRegression.over_sampling_adasyn import over_sampling_adasyn
from ImbalancedLearningRegression.dist_metrics import euclidean_dist, heom_dist, overlap_dist

## adaptive synthetic minority over-sampling technique for regression
def adasyn(

        ## main arguments / inputs
        data,  ## training set (pandas dataframe)
        y,  ## response variable y by name (string)
        k=5,  ## num of neighs for over-sampling (pos int)
        samp_method="balance",  ## oversampling ("balance" or extreme")
        drop_na_col=True,  ## auto drop columns with nan's (bool)
        drop_na_row=True,  ## auto drop rows with nan's (bool)

        ## phi relevance function arguments / inputs
        rel_thres=0.5,  ## relevance threshold considered rare (pos real)
        rel_method="auto",  ## relevance method ("auto" or "manual")
        rel_xtrm_type="both",  ## distribution focus ("high", "low", "both")
        rel_coef=1.5,  ## coefficient for box plot (pos real)
        rel_ctrl_pts_rg=None  ## input for "manual" rel method  (2d array)

):
    """
    the main function, designed to help solve the problem of imbalanced data
    for regression of Adasyn method, which applies over-sampling the minority
    class (rare values in a normal distribution of y, typically found at the tails)

    procedure begins with a series of pre-processing steps, and to ensure no
    missing values (nan's), sorts the values in the response variable y by
    ascending order, and fits a function 'phi' to y, corresponding phi values
    (between 0 and 1) are generated for each value in y, the phi values are
    then used to determine if an observation is either normal or rare by the
    threshold specified in the argument 'rel_thres'

    normal observations are placed into a majority class subset (normal bin),
    while rare observations are placed in a seperate minority class
    subset (rare bin) where they're over-sampled

    over-sampling is applied by a random sampling from the normal bin based
    on a calculated percentage control by the argument 'samp_method', if the
    specified input of 'samp_method' is "balance", less over-sampling is
    conducted, and if "extreme" is specified more over-sampling is conducted

    ADASYN represents the Adaptive Synthetic algorithm and oversamples the 
    minority class depending on an estimate of the local distribution of the class.

    'Adasyn' is only applied to numeric / continuous features, synthetic values
    found in nominal / categorical features, is generated by randomly selecting
    observed values found within their respective feature

    procedure concludes by post-processing and returns a modified pandas data
    frame containing over-sampled (adaptive synthetic) observations,
    the distribution of the response variable y should more appropriately
    reflect the minority class areas of interest in y that are under-
    represented in the original training set

    ref:

    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.

    He, H., Bai, Y., Garcia, E. A., & Li, S. (2008, June).
    ADASYN: Adaptive synthetic sampling approach for imbalanced
    learning. In 2008 IEEE international joint conference on neural
    networks (IEEE world congress on computational intelligence)
    (pp. 1322-1328). IEEE.
    https://www.ele.uri.edu/faculty/he/PDFfiles/adasyn.pdf.
    """

    ## pre-process missing values
    if bool(drop_na_col) == True:
        data = data.dropna(axis=1)  ## drop columns with nan's

    if bool(drop_na_row) == True:
        data = data.dropna(axis=0)  ## drop rows with nan's

    ## quality check for missing values in dataframe
    if data.isnull().values.any():
        raise ValueError("cannot proceed: data cannot contain NaN values")

    ## quality check for y
    if isinstance(y, str) is False:
        raise ValueError("cannot proceed: y must be a string")

    if y in data.columns.values is False:
        raise ValueError("cannot proceed: y must be an header name (string) \
               found in the dataframe")

    ## quality check for k number specification
    if k > len(data):
        raise ValueError("cannot proceed: k is greater than number of \
               observations / rows contained in the dataframe")

    ## quality check for sampling method
    if samp_method in ["balance", "extreme"] is False:
        raise ValueError("samp_method must be either: 'balance' or 'extreme' ")

    ## quality check for relevance threshold parameter
    if rel_thres == None:
        raise ValueError("cannot proceed: relevance threshold required")

    if rel_thres > 1 or rel_thres <= 0:
        raise ValueError("rel_thres must be a real number number: 0 < R < 1")


    ## store data dimensions
    n = len(data)
    d = len(data.columns)

    ## store original data types
    ## build a dataframe that contains d columns
    feat_dtypes_orig = [None] * d

    ## fill in the dataframe with original data types of each data in each column
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype

    ## determine column position for response variable y
    y_col = data.columns.get_loc(y)

    ## move response variable y to last column
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data = data[data.columns[cols]]

    ## store original feature headers and
    ## encode feature headers to index position
    feat_names = list(data.columns)
    data.columns = range(d)

    ## sort response variable y by ascending order
    y = pd.DataFrame(data[d - 1])
    y_sort = y.sort_values(by=d - 1)
    y_sort = y_sort[d - 1]

    ## -------------------------------- phi --------------------------------- ##
    ## calculate parameters for phi relevance function
    ## (see 'phi_ctrl_pts()' function for details)
    phi_params = phi_ctrl_pts(

        y=y_sort,  ## y (ascending)
        method=rel_method,  ## defaults "auto"
        xtrm_type=rel_xtrm_type,  ## defaults "both"
        coef=rel_coef,  ## defaults 1.5
        ctrl_pts=rel_ctrl_pts_rg  ## user spec
    )

    ## calculate the phi relevance function
    ## (see 'phi()' function for details)
    y_phi = phi(

        y=y_sort,  ## y (ascending)
        ctrl_pts=phi_params  ## from 'phi_ctrl_pts()'
    )

    ## label each observation
    ## if minority class - label 1, if majority class - label -1 # ????? Modified from Gloria and Lingyi's implementation
    # label = []
    # for i in range(0, len(y_sort)):
    #     if (y_phi[i] > rel_thres):
    #         label.append(1)
    #     else:
    #         label.append(-1)
    label = [0 for i in range(len(y_sort))]
    for i in range(len(y_sort)):
        if (y_phi[i] >= rel_thres):
            label[y_sort.index[i]] = 1
        else:
            label[y_sort.index[i]] = -1

    ## phi relevance quality check
    if all(i == 0 for i in y_phi):
        raise ValueError("redefine phi relevance function: all points are 1")

    if all(i == 1 for i in y_phi):
        raise ValueError("redefine phi relevance function: all points are 0")
    ## ---------------------------------------------------------------------- ##

    ## preprocess the entire dataset before entering the bump so that we only preprocess it once
    ## find non-negative numeric features
    feat_non_neg = []
    num_dtypes = ["int64", "float64"]

    for j in range(d):
        if data.iloc[:, j].dtype in num_dtypes and any(data.iloc[:, j] > 0):
            feat_non_neg.append(j)

    ## find features without variation (constant features)
    feat_const = data.columns[data.nunique() == 1]

    feat_var = []
    data_orig = data.copy()
    ## temporarily remove constant features
    if len(feat_const) > 0:

        ## create copy of orignal data and omit constant features
        data_orig = data.copy()
        data = data.drop(data.columns[feat_const], axis=1)

        ## store list of features with variation
        feat_var = list(data.columns.values)

        ## reindex features with variation
        for i in range(d - len(feat_const)):
            data.rename(columns={
                data.columns[i]: i
            }, inplace=True)

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

    # Unknown warning, may be handled later
    pd.options.mode.chained_assignment = None

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
    dist_matrix = np.ndarray(shape=(n, n))

    for i in tqdm(range(n), ascii=True, desc="dist_matrix"):
        for j in range(n):

            ## utilize euclidean distance given that
            ## data is all numeric / continuous
            if feat_count_nom == 0:
                dist_matrix[i][j] = euclidean_dist(
                    a=data_num.iloc[i],
                    b=data_num.iloc[j],
                    d=feat_count_num
                )

            ## utilize heom distance given that
            ## data contains both numeric / continuous
            ## and nominal / categorical
            if feat_count_nom > 0 and feat_count_num > 0:
                dist_matrix[i][j] = heom_dist(

                    ## numeric inputs
                    a_num=data_num.iloc[i],
                    b_num=data_num.iloc[j],
                    d_num=feat_count_num,
                    ranges_num=feat_ranges_num,

                    ## nominal inputs
                    a_nom=data_nom.iloc[i],
                    b_nom=data_nom.iloc[j],
                    d_nom=feat_count_nom
                )

            ## utilize hamming distance given that
            ## data is all nominal / categorical
            if feat_count_num == 0:
                dist_matrix[i][j] = overlap_dist(
                    a=data_nom.iloc[i],
                    b=data_nom.iloc[j],
                    d=feat_count_nom
                )

    ## determine indicies of k nearest neighbors
    knn_index = [None] * n

    for i in range(n):
        knn_index[i] = np.argsort(dist_matrix[i])
    
    ## end of preprocessing
    ## ---------------------------------------------------------------------- ##

    ## determine bin (rare or normal) by bump classification
    bumps = [0]

    for i in range(0, len(y_sort) - 1):
        if ((y_phi[i] >= rel_thres and y_phi[i + 1] < rel_thres) or
                (y_phi[i] < rel_thres and y_phi[i + 1] >= rel_thres)):
            bumps.append(i + 1)

    bumps.append(n)

    ## number of bump classes
    n_bumps = len(bumps) - 1

    ## num_of_bumps is an array that contains the number of samples in each
    ## minority or majority class
    num_in_bumps= []
    for i in range(n_bumps):
        num = bumps[i+1] - bumps[i]
        num_in_bumps.append(num)

    ## determine indicies for each bump classification
    b_index = {}

    for i in range(n_bumps):
        b_index.update({i: y_sort[bumps[i]:bumps[i + 1]]})

    ## calculate oversampling percentage according to
    ## bump class and user specified method ("balance" or "extreme")
    b = round(n / n_bumps)
    s_perc = []
    scale = []
    obj = []

    if samp_method == "balance":
        for i in b_index:
            s_perc.append(b / len(b_index[i]))

    if samp_method == "extreme":
        for i in b_index:
            scale.append(b ** 2 / len(b_index[i]))
        scale = n_bumps * b / sum(scale)

        for i in b_index:
            obj.append(round(b ** 2 / len(b_index[i]) * scale, 2))
            s_perc.append(round(obj[i] / len(b_index[i]), 1))

    ## conduct oversampling and store modified training set
    data_new = pd.DataFrame()


    for i in range(n_bumps):

        ## no sampling
        if s_perc[i] <= 1:
            ## simply return no sampling
            ## results to modified training set
            data_new = pd.concat([data_orig.iloc[b_index[i].index], data_new], ignore_index = True) # ????? verify if data_orig matches columns UPDATE: semi-verified

        ## over-sampling
        if s_perc[i] > 1:
            ## generate synthetic observations in training set
            ## considered 'minority'
            ## (see 'over_sampling()' function for details)
            synth_obs = over_sampling_adasyn(
                data = data,
                label = label,
                index = list(b_index[i].index),
                perc = s_perc[i],
                k = k,
                ## inputs from preprocessing
                knn_index = knn_index,
                feat_list_nom = feat_list_nom,
                feat_list_num = feat_list_num,
                feat_ranges  = feat_ranges,
                data_var = data_var,
                feat_const = feat_const,
                feat_var = feat_var,
                feat_non_neg = feat_non_neg,
                data_orig = data_orig
            )

            ## concatenate over-sampling
            ## results to modified training set
            data_new = pd.concat([synth_obs, data_new], ignore_index = True)

    ## rename feature headers to originals
    data_new.columns = feat_names

    ## restore response variable y to original position
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data_new = data_new[data_new.columns[cols]]

    ## restore original data types
    for j in range(d):
        data_new.iloc[:, j] = data_new.iloc[:, j].astype(feat_dtypes_orig[j])

    ## return modified training set
    return data_new
