from logging import Logger

import pandas as pd
import numpy as np
from catboost import Pool, CatBoost, CatBoostClassifier
from typing import *

from auction.constants import UNIQ_COL, TARGET_COL
from auction.splitter import BaseSplitter
from auction.utils import console_logger


class ModelInputs:
    """
    Class for managing dataset, make split and preparation
    """

    def __init__(self, data: pd.DataFrame, features_cols: List[str],
                 logger: Logger = console_logger(''), cat_features: List[str] = None, uniq_col: str = UNIQ_COL):
        """
        Init method
        :param data: path to self.data.csv
        :param features_cols: list of features columns names that model will be trained
        :param logger: logger to log self.data
        :param cat_features: list of categorical features
        """
        self.data = data
        self.data.sort_values([uniq_col], inplace=True)
        self.features_cols = features_cols
        self.logger = logger
        self.train: Optional[Pool] = None
        self.test: Optional[Pool] = None
        self.train_uniq_ids: np.ndarray = None
        self.test_uniq_ids: np.ndarray = None
        self.cat_features: List[str] = cat_features or []
        self.uniq_col = uniq_col
        self._validate_columns()

    def _validate_columns(self):
        """
        Make sure there are features cols in data
        :return:
        """
        missing_cat_features = set(self.cat_features).difference(self.features_cols)
        assert not missing_cat_features, \
            f'Categorical features {missing_cat_features} not in features_cols, features_cols are {self.features_cols}'

        missing_cols = set(self.features_cols + [self.uniq_col]).difference(self.data.columns)
        assert not missing_cols, \
            f'There are no {missing_cols} in data, features_cols are {self.features_cols},' \
            f' uniq_col: {self.uniq_col}'

    def make_test_train(self, splitter: BaseSplitter, target_col: str = TARGET_COL):
        """
        Make Pools of data to train and evaluate model
        :param splitter: Splitter object for train/test selection strategy
        :param target_col: target column name
        :return:
        """
        if self.cat_features:
            self.data.loc[:, self.cat_features] = self.data.loc[:, self.cat_features].astype(str)

        x = self.data.loc[:, self.features_cols]
        y = self.data.loc[:, target_col].values
        ids = self.data.loc[:, self.uniq_col].values.astype('str')

        # Split data
        self.train_uniq_ids, self.test_uniq_ids = splitter.split(self.data)

        pools_kwargs = {}
        for ds_name, uniq_ids in {'Train': self.train_uniq_ids, 'Test': self.test_uniq_ids}.items():
            indices = np.isin(ids, uniq_ids)
            pools_kwargs[ds_name] = {
                'data': x.loc[indices, :],
                'label': y[indices],
            }
            self.logger.info(f'{ds_name} data shape is {pools_kwargs[ds_name]["data"].shape}')

            if self.cat_features:
                pools_kwargs[ds_name]['cat_features'] = self.cat_features

        self.train = Pool(**pools_kwargs['Train'])
        self.test = Pool(**pools_kwargs['Test'])

        self.logger.warning(f' Test uniq targets: {len(set(pools_kwargs["Test"]["label"]))};'
                            f' Train uniq targets: {len(set(pools_kwargs["Train"]["label"]))};')