import logging
import itertools
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from preprocessData import preprocessData
from computeFeatures import computeFeatures
from util import createSplits

# Initialize logging
logging.basicConfig(filename='model.log', level=logging.INFO)

# Preprocess data once, outside of loop
df_sensor, df_smell = preprocessData(in_p=["dataset/esdr_raw/", "dataset/smell_raw.csv"])
wanted_cols = ["DateTime", "3.feed_24.PM10_UG_M3", "3.feed_27.CO_PPB", "3.feed_28.SONICWS_MPH", "3.feed_26.SONICWS_MPH"]
df_sensor = df_sensor[wanted_cols]

# Initialize
best_f1 = 0
best_model = None
best_params = {}
results_list = []

# Hyperparameter & Model ranges
hyperparams = {
    'smell_thr_values': [30, 40, 50],
    'smell_predict_hrs_values': [4, 6, 8],
    'look_back_hrs_values': [1, 2, 3],
    'add_inter_values': [False],
    'test_size_values': [2680, 3000],
    'train_size_values': [6360, 7000]
}
models = [DummyClassifier(), DecisionTreeClassifier(), RandomForestClassifier(n_jobs=-1), MLPClassifier()]

# Multiple Scoring Metrics
multiple_scorers = {'precision': make_scorer(precision_score, zero_division=0), 
                    'recall': make_scorer(recall_score),
                    'f1_score': make_scorer(f1_score), 
                    'accuracy': make_scorer(accuracy_score)}

# Main loop
for hyperparam_combo in itertools.product(*hyperparams.values()):
    params = dict(zip(hyperparams.keys(), hyperparam_combo))

    df_X, df_Y, _ = computeFeatures(df_esdr=df_sensor, df_smell=df_smell,
                                    f_hr=params['smell_predict_hrs_values'],
                                    b_hr=params['look_back_hrs_values'],
                                    thr=params['smell_thr_values'],
                                    add_inter=params['add_inter_values'])

    splits = createSplits(params['test_size_values'], params['train_size_values'], df_X.shape[0])

    for model in models:
        result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=multiple_scorers, n_jobs=-1)
        new_f1 = result['test_f1_score'].mean()

        # Logging results
        logging.info(f"Model: {model}, Metrics: {result.mean()}, Params: {params}")

        results_list.append({'model': model, 'Metrics': result, 'params': params})

        if new_f1 > best_f1:
            best_f1 = new_f1
            best_model = model
            best_params = params

# Sort and log all results by F1 score
results_list = sorted(results_list, key=lambda x: x['Metrics']['test_f1_score'].mean(), reverse=True)
logging.info(f"Best model: {best_model}, Best Metrics: {best_f1}")
logging.info(f"Best hyperparameters: {best_params}")

print(f"Best model: {best_model}, Best F1 Score: {best_f1}")
print(f"Best hyperparameters: {best_params}")
