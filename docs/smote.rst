SMOTE
========================================================

Synthetic Minority Oversampling Technique.
SMOTE is an over-sampling method that synthesizes new samples by using one of the neighbors of a seed sample.

.. py:function:: smote(data, y, k = 5, samp_method = "balance", drop_na_col = True, drop_na_row = True, rel_thres = 0.5, rel_method = "auto", rel_xtrm_type = "both", rel_coef = 1.5, rel_ctrl_pts_rg = None)
   
   :param data: Pandas dataframe, the dataset to re-sample.
   :type data: :term:`Pandas dataframe`
   :param str y: Column name of the target variable in the Pandas dataframe.
   :param int k: The number of neighbors considered. Must be a positive integer.
   :param str samp_method: Method to determine re-sampling percentage. Either ``balance`` or ``extreme``.
   :param bool drop_na_col: Determine whether or not automatically drop columns containing NaN values. The data frame should not contain any missing values, so it is suggested to keep it as default.
   :param bool drop_na_row: Determine whether or not automatically drop rows containing NaN values. The data frame should not contain any missing values, so it is suggested to keep it as default.
   :param float rel_thres: Relevance threshold, above which a sample is considered rare. Must be a real number between 0 and 1 (0, 1].
   :param str rel_method: Method to define the relevance function, either ``auto`` or ``manual``. If ``manual``, must specify ``rel_ctrl_pts_rg``.
   :param str rel_xtrm_type: Distribution focus, ``high``, ``low``, or ``both``. If ``high``, rare cases having small y values will be considerd as normal, and vise versa.
   :param float rel_coef: Coefficient for box plot.
   :param rel_ctrl_pts_rg: Manually specify the regions of interest. See `SMOGN advanced example <https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_3_adv.ipynb>`_ for more details.
   :type rel_ctrl_pts_rg: :term:`2D array`
   :return: Re-sampled dataset.
   :rtype: :term:`Pandas dataframe`
   :raises ValueError: If an input attribute has wrong data type or invalid value, or relevance values are all zero or all one, or synthetic data contains missing values.

References
----------
[1] L. Torgo, R. P. Ribeiro, B. Pfahringer, P. Branco, “Smote for regression,” In Portuguese conference on artificial intelligence, pp. 378-389, 2013. Springer, Berlin, Heidelberg.

[2] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research, 321-357, 2002.

Examples
--------
.. doctest::

    >>> from ImbalancedLearningRegression import smote
    >>> housing = pandas.read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv")
    >>> housing_smote = smote(data = housing, y = "SalePrice")
