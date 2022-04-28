ADASYN
========================================================

ADASYN is an over-sampling method that is similar to SMOTE, but more samples are generated for seed samples that are difficult to be learned.

.. py:function:: adasyn(data, y, k = 5, samp_method = "balance", drop_na_col = True, drop_na_row = True, rel_thres = 0.5, rel_method = "auto", rel_xtrm_type = "both", rel_coef = 1.5, rel_ctrl_pts_rg = None)
   
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
   :raises AssertionError: If normalized ratio ri does not sum up to one.

References
----------
[1] He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. “ADASYN: Adaptive synthetic sampling approach for imbalanced learning,” In IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), pp. 1322-1328, 2008.

Examples
--------
.. doctest::

    >>> from ImbalancedLearningRegression import adasyn
    >>> housing = pandas.read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv")
    >>> housing_adasyn = adasyn(data = housing, y = "SalePrice")
