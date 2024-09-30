from pathlib import Path

import pandas as pd

from auction.constants import TARGET_COL, UNIQ_COL
from auction.inputs import ModelInputs
from auction.model import CatModel
from auction.splitter import RandomSplitter
from auction.utils import add_uniq_index_for_missing_values, parse_train_report

"""
Script for the submission data creation
"""

def main():

    train = pd.read_csv('train_data.csv')
    test = pd.read_csv('test_data.csv')

    exp_tag = 'baseline_without_const_feats_ifa_sampling_fill_na_ifas'

    train_dir = Path(f'./model_logs/{exp_tag}')
    train_dir.mkdir(parents=True, exist_ok=True)

    # I have removed features with one and only unique feature value across train and test data
    features_to_skip = [
        'ssp', 'sdk', 'adt', 'dc',
        TARGET_COL, UNIQ_COL,
    ]

    feature_cols = [x for x in train.columns if x not in features_to_skip]

    cat_cols = [
        'dsp',
        'auctionBidFloorSource',
        'sdkver',
        'bundle',
        'os',
        'lang',
        'country',
        'region',
        'bidderFlrPolicy',
        'contype',
        'request_context_device_type',
    ]

    # Fill na values of ifa with fake_values
    train = add_uniq_index_for_missing_values(train, 'ifa')

    # We use only train data for the next experiments
    inputs = ModelInputs(data=train, features_cols=feature_cols, cat_features=cat_cols)

    # Split Train dataset randomly for model evaluation (70% goes to train 30% goes to validation);
    # We sample data by IFA; One unique value goes only to one sample type (train/test)
    params = {
        "train_test_ratio": 0.70,
        "seed": 1,
        "uniq_col": "ifa"
    }

    splitter = RandomSplitter(params=params)
    inputs.make_test_train(splitter)

    # Create CatBoost model classifier
    params = {
        "train_dir": str(train_dir),
        "loss_function": "Logloss",
        "iterations": 200,
        "custom_metric": [
            "F1:hints=skip_train~false", "AUC:hints=skip_train~false",
        ],
        "random_seed": 0,
        "metric_period": 15,
        "depth": 4,
        "task_type": "GPU",
        "devices": "1",
        "learning_rate": 0.05
    }

    model = CatModel(params)

    # Fit model
    model.fit(inputs)
    # Plot importance on validation set
    model.plot_feature_importance(inputs)

    # Print training progress (train and validation sample metrics)
    print(parse_train_report(exp_tag).to_mardown())

    # Compute validation metrics on Train data and Submission (Test dataset)
    na_value = -99999
    model.evaluate_model(train.fillna(na_value), test.fillna(na_value))

    prediction = model.predict(test.fillna(na_value))
    df = pd.DataFrame({'prediction': prediction})
    df.to_csv('submission.csv')

if __name__ == '__main__':
    main()
