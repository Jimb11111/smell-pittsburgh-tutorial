import logging
import itertools
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from preprocessData import preprocessData
from computeFeatures import computeFeatures  # Modify this function according to the guide
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
    'smell_thr_values': [15, 20, 25, 30],
    'smell_predict_hrs_values': [4, 6, 8, 10],
    'look_back_hrs_values': [1, 5, 8, 10],
    'add_inter_values': [False],
    'test_size_values': [1000, 3000],
    'train_size_values': [4000, 7000]
}
models = [DummyClassifier(), DecisionTreeClassifier(), RandomForestClassifier(n_jobs=-1), MLPClassifier()]

# Multiple Scoring Metrics
multiple_scorers = {'precision': make_scorer(precision_score, zero_division=0), 
                    'recall': make_scorer(recall_score),
                    'f1_score': make_scorer(f1_score), 
                    'accuracy': make_scorer(accuracy_score)}

# Counters for progress tracking
total_combinations = len(list(itertools.product(*hyperparams.values())))
combo_count = 0

# Main loop
for hyperparam_combo in itertools.product(*hyperparams.values()):
    combo_count += 1
    params = dict(zip(hyperparams.keys(), hyperparam_combo))

    # Modify computeFeatures function for this to work
    df_X, df_Y, _ = computeFeatures(df_esdr=df_sensor, df_smell=df_smell,
                                    f_hr=params['smell_predict_hrs_values'],
                                    b_hr=params['look_back_hrs_values'],
                                    thr=params['smell_thr_values'],
                                    add_inter=params['add_inter_values'])
    
    # Display progress for hyperparameters
    print(f"Processed hyperparam sets: {combo_count}/{total_combinations}")

    splits = createSplits(params['test_size_values'], params['train_size_values'], df_X.shape[0])

    model_count = 0
    total_models = len(models)

    for model in models:
        model_count += 1
        result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=multiple_scorers, n_jobs=-1)
        new_f1 = result['test_f1_score'].mean()

        mean_metrics = {key: val.mean() for key, val in result.items() if isinstance(val, np.ndarray)}

        logging.info(f"Model: {model}, Metrics: {mean_metrics}, Params: {params}")
        
        results_list.append({'model': model, 'Metrics': mean_metrics, 'params': params})

        if new_f1 > best_f1:
            best_f1 = new_f1
            best_model = model
            best_params = params
        
        # Display progress for models
        print(f"Processed models: {model_count}/{total_models}")

results_list = sorted(results_list, key=lambda x: x['Metrics']['test_f1_score'], reverse=True)
logging.info(f"Best model: {best_model}, Best Metrics: {best_f1}")
logging.info(f"Best hyperparameters: {best_params}")

print(f"Best model: {best_model}, Best F1 Score: {best_f1}")
print(f"Best hyperparameters: {best_params}")
