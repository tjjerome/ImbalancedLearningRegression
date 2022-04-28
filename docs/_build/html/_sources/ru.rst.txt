Random Under-sampling
========================================================

Random Under-sampling is an under-sampling method that randomly select a subset of majority samples.

.. py:function:: random_under(data, y, samp_method = "balance", drop_na_col = True, drop_na_row = True, replacement = False, manual_perc = False, perc_u = -1, rel_thres = 0.5, rel_method = "auto", rel_xtrm_type = "both", rel_coef = 1.5, rel_ctrl_pts_rg = None)
   
   :param data: Pandas dataframe, the dataset to re-sample.
   :type data: :term:`Pandas dataframe`
   :param str y: Column name of the target variable in the Pandas dataframe.
   :param str samp_method: Method to determine re-sampling percentage. Either ``balance`` or ``extreme``.
   :param bool drop_na_col: Determine whether or not automatically drop columns containing NaN values. The data frame should not contain any missing values, so it is suggested to keep it as default.
   :param bool drop_na_row: Determine whether or not automatically drop rows containing NaN values. The data frame should not contain any missing values, so it is suggested to keep it as default.
   :param bool replacement: Randomly select sample to duplicate: with or without replacement.
   :param bool manual_perc: Keep the same percentage of re-sampling for all bins. If ``True``, ``perc_o`` is required to be a positive real number.
   :param float perc_u: User-specified fixed percentage of under-sampling for all bins. Must be a real number between 0 and 1 (0, 1) if ``manual_perc = True``.
   :param float rel_thres: Relevance threshold, above which a sample is considered rare. Must be a real number between 0 and 1 (0, 1].
   :param str rel_method: Method to define the relevance function, either ``auto`` or ``manual``. If ``manual``, must specify ``rel_ctrl_pts_rg``.
   :param str rel_xtrm_type: Distribution focus, ``high``, ``low``, or ``both``. If ``high``, rare cases having small y values will be considerd as normal, and vise versa.
   :param float rel_coef: Coefficient for box plot.
   :param rel_ctrl_pts_rg: Manually specify the regions of interest. See `SMOGN advanced example <https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_3_adv.ipynb>`_ for more details.
   :type rel_ctrl_pts_rg: :term:`2D array`
   :return: Re-sampled dataset.
   :rtype: :term:`Pandas dataframe`
   :raises ValueError: If an input attribute has wrong data type or invalid value, or relevance values are all zero or all one, or under_sampled data contains missing values.

Examples
--------
.. doctest::

    >>> from ImbalancedLearningRegression import random_under
    >>> housing = pandas.read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv")
    >>> housing_ru = random_under(data = housing, y = "SalePrice")
