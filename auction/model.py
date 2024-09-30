from abc import abstractmethod
from logging import Logger
from typing import List, Dict, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

from auction.base import StrategyAuction
from auction.constants import TARGET_COL
from auction.inputs import ModelInputs
from auction.utils import console_logger


class BaseModel(StrategyAuction):
    """
    Base class of model to be used in auction
    """
    def __init__(self, params: Dict, logger: Logger = console_logger('')):
        """
        Init method
        :param params: parameters dict for data splitter
        :param logger: logger
        """
        super().__init__(params=params, logger=logger)

    @abstractmethod
    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """

        :param data: input dataframe with features cols
        :param kwargs: pickle kwargs
        :return: scores 1d array len of data.shape[0]
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, inputs: ModelInputs):
        """
        Fit model with some input data
        :param inputs: ModelInputs
        :return:
        """
        raise NotImplementedError


class CatModel(BaseModel):
    """Catboost model"""

    def __init__(self, params: Dict, logger: Logger = console_logger('')):
        """
        Init method
        :param params: dict with params for catboost model
        :param logger: logger
        """
        super().__init__(params=params, logger=logger)
        self.cat = CatBoostClassifier(**params)
        self.features_cols: List[str] = None

    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict scores for some data
        :param data: dataframe of features
        :return: array of scores in respect to data order
        """
        return self.cat.predict(data.loc[:, self.features_cols])

    def fit(self, inputs: ModelInputs):
        """
        Train model on data
        :param inputs: inputs object
        :return:
        """
        self.features_cols = inputs.features_cols
        self.cat.fit(inputs.train, plot=True, eval_set=inputs.test)

    def plot_feature_importance(
            self,
            inputs: ModelInputs,
            plots: Sequence[str] = ('PredictionValuesChange', 'LossFunctionChange', 'Interaction')) -> plt.Figure:
        """
        Plots CatBoost built-in feature importance bar-plots.
        :param model: CatModel.
        :param inputs: ModelInputs, input data of model.
        :param plots: sequence of str, names of feature importance plots to draw.
        :return: plt.Figure, feature importance bar-plots.
        """
        feature_importances = {
            fi_type: self.cat.get_feature_importance(inputs.test, type=fi_type)
            for fi_type in plots
        }
        heights = [len(fi) for _, fi in feature_importances.items()]
        fig, axes = plt.subplots(len(plots), 1, figsize=(10, .3 * sum(heights)),
                                 gridspec_kw={'height_ratios': heights})

        for fi_type, ax in zip(plots, axes):
            fnames = self.cat.feature_names_
            fi = feature_importances[fi_type]
            if fi_type == 'Interaction':
                fnames = [f'{fnames[i]} & {fnames[j]}' for i, j in fi[:, :2].astype(int)]
                fi = fi[:, 2]
            df_fi = pd.DataFrame({'feature': fnames, 'score': fi}).sort_values(by='score')

            ax = df_fi.plot('feature', 'score', kind='barh', ax=ax)
            ax.grid(axis='x')
            ax.yaxis.tick_right()
            ax.set_title(f'Feature importance ({fi_type})')
            ax.set_yticklabels(ax.get_yticklabels(), ha='right')


    def evaluate_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Print F1 and AUC for train and test dataset
        :param train_df: train pandas dataframe
        :param test_df: test pandas dataframe
        :return:
        """
        test_f1 = f1_score(self.predict(test_df), test_df.loc[:, TARGET_COL])
        train_f1 = f1_score(self.predict(train_df), train_df.loc[:, TARGET_COL])

        test_auc = roc_auc_score(self.predict(test_df), test_df.loc[:, TARGET_COL])
        train_auc = roc_auc_score(self.predict(train_df), train_df.loc[:, TARGET_COL])

        self.logger.info(f"F1:  Train: {train_f1}; Test: {test_f1}")
        self.logger.info(f"AUC: Train: {train_auc}; Test: {test_auc}")
