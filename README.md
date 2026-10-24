# WIDS_DATATHON_4THPLACE_2026
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')
!pip install lightgbm scikit-learn==1.8.0 scikit-survival==0.27.0 --quiet
**Import Libraries**
#import numpy as np
#import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
The script begins by importing the libraries required for data manipulation, machine learning, survival analysis, and model evaluation. 
*NumPy* is used for numerical computations, while *Pandas* handles data loading and preprocessing. *LightGBM* provides efficient gradient boosting classifiers for the horizon-specific binary prediction tasks. From *scikit-learn*, *StratifiedKFold* is imported to perform stratified cross-validation, ensuring a balanced distribution of event occurrences across folds. Finally, *scikit-survival* supplies the survival data structure (*Surv*) and the *GradientBoostingSurvivalAnalysis* model used for time-to-event prediction.
**Global Configuration**
SEEDS = [42, 2024, 777]   #reduced for stability
FOLDS = 5

HORIZONS = [12, 24, 48, 72]

W = {12: (0.97, 0.03),
    24: (0.95, 0.05),
    48: (0.45, 0.55)}

POWER_24 = 1.1   #mild calibration (safe)
Several constants are defined at the beginning of the script to control the training procedure. Multiple random seeds are specified to reduce prediction variance and improve robustness through model averaging. The number of cross-validation folds is fixed at 5 to balance training stability and computational cost. Prediction horizons (12h, 24h, 48h, and 72h) define the evaluation time points at which event probabilities are estimated. Additional parameters, such as blending weights and calibration factors, are introduced to fine-tune the contribution of each model and improve probability calibration.
**Data Loading**
train = pd.read_csv("/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26/train.csv")
test  = pd.read_csv("/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26/test.csv")
The training and test datasets are loaded from CSV files using Pandas. The training dataset contains both predictor variables and target information (event and time_to_hit_hours), while the test dataset contains only the predictors required for inference. Loading the data at the start of the pipeline ensures that all subsequent preprocessing and modeling steps operate on a consistent dataset structure.
**Feature Engineering**
#%% FEATURE ENGINEERING

def create_features(df):
    df = df.copy()

    dist = df['dist_min_ci_0_5h'].clip(1)
    speed = df['closing_speed_m_per_h']

    # --- distance ---
    df['dist_km'] = dist / 1000
    df['log_dist'] = np.log1p(dist)
    df['inv_dist'] = 1/(df['dist_km']+0.1)

    # --- fire geometry ---
    fire_radius = np.sqrt(df['area_first_ha'] * 10000 / np.pi)
    df['radius_dist_ratio'] = fire_radius / dist

    # --- movement ---
    safe_speed = speed.clip(lower=0)
    df['eta'] = dist / (safe_speed + 1e-5)

    effective_speed = safe_speed + df['radial_growth_rate_m_per_h'].clip(lower=0)
    df['eta_effective'] = dist / (effective_speed + 1)

    # --- threat ---
    df['threat'] = df['alignment_abs'] * speed / np.log1p(dist)

    # --- growth ---
    df['growth'] = df['area_growth_rate_ha_per_h'] * df['num_perimeters_0_5h']

    # --- temporal ---
    df['hour_sin'] = np.sin(2*np.pi*df['event_start_hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['event_start_hour']/24)

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df

train = create_features(train)
test  = create_features(test)
This section creates additional variables from the original dataset to capture relationships between fire behavior, distance, movement, and temporal patterns. Features such as logarithmic distance, inverse distance, estimated time-to-arrival (ETA), fire radius-to-distance ratio, and threat indicators are engineered to provide the models with more informative predictors than the raw inputs alone. Cyclical time features (hour_sin, hour_cos) are also added to represent the hour of the day while preserving its periodic nature. Missing values and infinite values are replaced to ensure model stability.
**Data Preparation**
#%% DATA PREP

X = train.drop(columns=['event_id','event','time_to_hit_hours'])
X_test = test.drop(columns=['event_id'])

y_surv = Surv.from_arrays(
    event=train['event'].astype(bool),
    time=train['time_to_hit_hours']
)
The training and test datasets are transformed using the feature engineering pipeline. Predictor variables (X) are separated from identifiers and target columns, while the survival target is converted into the structured format required by scikit-survival. This representation combines the event indicator (whether the fire reached the asset) and the corresponding time-to-event information.
**Helpers**
#%% HELPERS

def get_surv_preds(model, X):
    surv = model.predict_survival_function(X)
    preds = np.zeros((len(surv), 4))

    for i, fn in enumerate(surv):
        t_min, t_max = fn.domain
        
        # CRUCIAL CLIPPING
        safe_times = np.clip(HORIZONS, t_min, t_max)
        
        preds[i,:] = 1 - fn(safe_times)

    return preds

def enforce_monotonicity(p):
    for i in range(1,4):
        p[:,i] = np.maximum(p[:,i], p[:,i-1])
    return p

def make_target(time, event, h):
    return ((event==1) & (time<=h)).astype(int)
Several utility functions are defined to simplify the workflow. The get_surv_preds() function converts survival curves into cumulative event probabilities at the required prediction horizons (12h, 24h, 48h, and 72h). enforce_monotonicity() ensures that predicted probabilities increase over time, which is consistent with survival analysis principles. Finally, make_target() creates binary classification targets for each prediction horizon, indicating whether the event occurred before a specified time threshold.
**OPTIONAL: Measure the Time it Takes to Train the Model**
import time as timer

start = timer.time()
#the end of the function is pointed near the end of the algorithm
**GradientBoostingSurvivalAnalysis Survival Core**
#%% GBSA (SURVIVAL CORE)

gbsa_cfgs = [
    dict(n_estimators=1200, learning_rate=0.01, max_depth=3),
    dict(n_estimators=1500, learning_rate=0.008, max_depth=2),
    dict(n_estimators=1000, learning_rate=0.015, max_depth=3),
]

kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_gbsa = np.zeros((len(X),4))
test_gbsa = np.zeros((len(X_test),4))

for cfg in gbsa_cfgs:
    for seed in SEEDS:

        fold_oof = np.zeros((len(X),4))
        fold_test = np.zeros((len(X_test),4))

        for tr,va in kf.split(X, train['event']):

            model = GradientBoostingSurvivalAnalysis(**cfg, random_state=seed)

            model.fit(X.iloc[tr], y_surv[tr])

            fold_oof[va] = get_surv_preds(model, X.iloc[va])
            fold_test   += get_surv_preds(model, X_test) / FOLDS

        oof_gbsa += fold_oof / (len(SEEDS)*len(gbsa_cfgs))
        test_gbsa += fold_test / (len(SEEDS)*len(gbsa_cfgs))
This is the core survival modeling component. Multiple GradientBoostingSurvivalAnalysis configurations are trained using stratified cross-validation and several random seeds to improve robustness. Each model learns the time-to-event distribution and generates survival probabilities for the predefined horizons. Predictions from all configurations and seeds are averaged to reduce variance and improve generalization.
**LightGBM Classification Models**
#%% LGB (IPCW-lite but stable)

oof_lgb = {}
test_lgb = {}

for h in [12,24,48]:

    y = make_target(train['time_to_hit_hours'], train['event'], h)

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:

        for tr,va in kf.split(X, train['event']):

            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=seed
            )

            model.fit(X.iloc[tr], y.iloc[tr])

            oof[va] += model.predict_proba(X.iloc[va])[:,1] / len(SEEDS)
            preds   += model.predict_proba(X_test)[:,1] / (len(SEEDS)*FOLDS)

    oof_lgb[h] = oof
    test_lgb[h] = preds
In addition to the survival model, separate binary LightGBM classifiers are trained for the 12h, 24h, and 48h horizons. For each horizon, the task is converted into a binary classification problem that predicts whether the event occurs before the specified time limit. Cross-validation and multiple random seeds are again used to produce stable out-of-fold and test predictions.
**Model Blending and Calibration**
#%% BLEND (SAFE + STRONG)

# calibration only for 24h (proven)
oof_gbsa[:,1] = np.clip(oof_gbsa[:,1] ** POWER_24, 0, 1)
test_gbsa[:,1] = np.clip(test_gbsa[:,1] ** POWER_24, 0, 1)

oof_final = oof_gbsa.copy()
test_final = test_gbsa.copy()

for i,h in enumerate([12,24,48]):
    w_gbsa, w_lgb = W[h]
    oof_final[:,i]  = w_gbsa*oof_gbsa[:,i] + w_lgb*oof_lgb[h]
    test_final[:,i] = w_gbsa*test_gbsa[:,i] + w_lgb*test_lgb[h]

# 72h trick (VERY IMPORTANT)
oof_final[:,3]  = 0.9
test_final[:,3] = 0.9

oof_final  = enforce_monotonicity(oof_final)
test_final = enforce_monotonicity(test_final)

The final predictions are obtained by combining the strengths of the survival model and the LightGBM classifiers. Horizon-specific weights control the contribution of each model type. A mild calibration step is applied to the 24-hour survival predictions to improve probability calibration. The 72-hour probability is fixed at a high constant value, reflecting the assumption that most events occur within this extended time window. Monotonicity is enforced after blending to ensure logically consistent probabilities across horizons.
**Anti-Overfit Check**
#%% SANITY CHECK (ANTI-OVERFIT)

print("OOF stats:")
print("mean:", oof_final.mean(axis=0))
print("std :", oof_final.std(axis=0))
Before generating the final submission, summary statistics of the out-of-fold predictions are printed. The mean and standard deviation of predicted probabilities provide a quick validation that the model outputs are reasonable and do not exhibit obvious signs of overfitting or instability.
**SUBMISSION**
#%% SUBMISSION

submission = pd.DataFrame({
    'event_id': test['event_id'],
    'prob_12h': test_final[:,0],
    'prob_24h': test_final[:,1],
    'prob_48h': test_final[:,2],
    'prob_72h': test_final[:,3],
})

submission.to_csv("submission.csv", index=False)
print("Saved as <<submission.csv>>")
The final predicted probabilities for all horizons are combined with the corresponding event_id values from the test set. The resulting submission file is saved in CSV format and is ready for evaluation or competition submission.
**OPTIONAL: end timer for model**
end = timer.time()
print(f"Total time to train: {(end-start)/60:.1f} minutes")
