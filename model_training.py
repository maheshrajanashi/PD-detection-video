import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import random
import click
import imblearn

from tqdm import tqdm
from pandas import DataFrame
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from shaphypetune import BoostSearch, BoostRFE, BoostRFA, BoostBoruta
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, r2_score, mean_absolute_percentage_error, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

from feature_extraction import extract_features

RATING_FILE = "./severity_dataset_dropped_correlated_columns.csv"

def load():
    '''
    Return three arrays: features, labels, patient_ids (features[i], labels[i], and patient_ids[i] should correspond to the same data point)
    This paper uses leave-one-patient-out cross validation, which requires grouping data based on patient IDs.
    If your dataset does not have associated patient IDs, you will need to re-implement the evaluation method, as leave-one-patient-out may not be applicable.
    '''
    df = pd.read_csv(RATING_FILE)
    # group by hand
    features = df.loc[:, 'wrist_mvmnt_x_median':'acceleration_min_trimmed']
    labels = df["Rating"]

    def parse(name:str):
        if name.startswith("NIH"): [ID, *_] = name.split("-")
        else: [*_, ID, _, _] = name.split("-")
        return ID
    
    df["id"] = df.filename.apply(parse)

    return features, labels, df["id"]

def select(features:DataFrame, labels, **cfg):
    '''
    Rank the features based on a "base model", 
    return top-n features where n is a hyper-parameter
    '''
    methods = { "BoostRFE":BoostRFE, "BoostRFA":BoostRFA, "BoostBoruta":BoostBoruta }

    SELECTOR = methods[cfg["selector"]]

    base = XGBRegressor() if cfg["selector_base"] == "XGB" else LGBMRegressor()
    
    selector = SELECTOR(base)
    selector.fit(features, labels)

    sorts = selector.ranking_.argsort()
    selected = features.columns[sorts][:cfg["n"]]
    features = features[selected]

    return features, labels


def metrics(preds, labels):
    results = {}

    preds, labels = onp.array(preds), onp.array(labels)
    results["mae"] = mean_absolute_error(labels, preds)
    results["mse"] = mean_squared_error(labels, preds)
    results["r2"] = r2_score(labels, preds)
    results["mape"] = mean_absolute_percentage_error(labels + 1, preds + 1) # shift labels by 1
    results["pearsonr"], _ = stats.pearsonr(labels, preds)

    rounded_preds = onp.round(preds)
    rounded_labels = onp.round(labels)
    results["kappa.no.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights=None)
    results["kappa.linear.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights="linear")
    results["kappa.quadratic.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights="quadratic")
    results["accuracy"] = accuracy_score(rounded_labels, rounded_preds)
    results["kenndalltau"], _ = stats.kendalltau(labels, preds)
    results["spearmanr"], _ = stats.spearmanr(labels, preds)

    return results


def model(**cfg):
    if cfg["model"] == "AdaBoostRegressor":
        return AdaBoostRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            random_state=cfg["random_state"],
        )

    if cfg["model"] == "RandomForestRegressor":
        return RandomForestRegressor(
            max_depth=cfg["max_depth"],
            max_features=cfg["max_features"],
            n_estimators=cfg["n_estimators"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=cfg["random_state"],
        )

    if cfg["model"] == "SVR":
        return SVR(C=cfg["c"], gamma=cfg["gamma"])

    if cfg["model"] == "LGBMRegressor":
        return LGBMRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            subsample=cfg["subsample"],
            random_state=cfg["random_state"],
        )

    if cfg["model"] == "XGBRegressor":
        return XGBRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            random_state=cfg["random_state"]
        )

    raise ValueError("Unknown model")



@click.command()
@click.option("--model", default="LGBMRegressor", help="Model to use")
@click.option("--selector", default="BoostRFE", help="Feature selection method")
@click.option("--selector_base", default="LGBM", help="Base regressor for feature selection")
@click.option("--n_estimators", default=611, help="Number of estimators for regressor")
@click.option("--learning_rate", default=0.01313, help="Learning rate for regressor")
@click.option("--max_depth", default=3, help="Max depth for regressor")
@click.option("--colsample_bytree", default=0.8, help="Colsample by tree for regressor")
@click.option("--subsample", default=0.8, help="Subsample for regressor")
@click.option("--reg_alpha", default=0.1, help="Reg alpha for regressor")
@click.option("--reg_lambda", default=0.1, help="Reg lambda for regressor")
@click.option("--max_features", default="sqrt", help="Max features for regressor")
@click.option("--min_samples_split", default=2, help="Min samples split for regressor")
@click.option("--min_samples_leaf", default=1, help="Min samples leaf for regressor")
@click.option("--min_child_weight", default=1, help="Min child weight for regressor")
@click.option("--C", default=1.0, help="C for regressor")
@click.option("--gamma", default="scale", help="Gamma for regressor")
@click.option("--n", default=22, help="Number of features to select")
@click.option("--random_state", default=42, help="Random state for regressor")
@click.option("--seed", default=42, help="Seed for random")
@click.option("--use_feature_selection",default='yes',help="yes if you want to select features, no if you want to work with all features")
@click.option("--use_feature_scaling",default='yes',help="yes if you want to scale the features, no otherwise")
@click.option("--scaling_method",default='StandardScaler',help="Options: StandardScaler, MinMaxScaler")
@click.option("--minority_oversample",default='no',help="Options: yes, no")
def main(**cfg):
    '''
    If you do not want wandb setup, comment all the lines starting with "wandb."
    If using wandb, create a project and use the appropriate project name.
    '''
    # wandb.init(project="npj-severity-paper", config=cfg)
    features, labels, ids = load()

    if(cfg["use_feature_selection"]=='yes'):
        features, labels = select(features, labels, **cfg)
    
    regressor = model(**cfg)


    oversample = SMOTE(random_state = cfg['random_state'])



    print("Training the super model on the entire dataset...")
    if cfg['use_feature_scaling'] == 'yes':
        if cfg['scaling_method'] == 'StandardScaler':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        features = scaler.fit_transform(features)

    if cfg['minority_oversample'] == 'yes':
        (features, labels) = oversample.fit_resample(features, labels)

    # Train the model on all data
    regressor.fit(features, labels)

    # Save the super model
    super_model_filename = "super_model.joblib"
    joblib.dump(regressor, super_model_filename)
    print(f"Super model saved to {super_model_filename}")

    

if __name__ == "__main__":
    main()
