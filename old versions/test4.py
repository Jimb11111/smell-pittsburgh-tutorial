import itertools

def pretty_print(df, message):
    print("\n================================================")
    print(f"{message}\n")
    print(df)
    print("\nColumn names below:")
    print(list(df.columns))
    print("================================================\n")

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

from preprocessData import preprocessData
from computeFeatures import computeFeatures
from util import scorer, printScores, createSplits, computeFeatureImportance

# Preprocess data once, outside of loop
df_sensor, df_smell = preprocessData(in_p=["dataset/esdr_raw/", "dataset/smell_raw.csv"])
pretty_print(df_sensor, "Display all sensor data and column names")
pretty_print(df_smell, "Display smell data and column names")

# Initialize
best_f1 = 0
best_model = None
best_params = {}
results_list = []

# Hyperparameter & Model ranges
hyperparams = {
    'smell_thr_values': [30, 50],
    'smell_predict_hrs_values': [4, 8],
    'look_back_hrs_values': [1, 3],
    'add_inter_values': [True, False],
    'test_size_values': [2680, 3000],
    'train_size_values': [6360, 7000]
}
models = [DummyClassifier(), DecisionTreeClassifier(), RandomForestClassifier(n_jobs=-1), MLPClassifier()]

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
        result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer, n_jobs=-1)
        new_f1 = result['test_f1'].mean()
        
        # Storing results
        results_list.append({'model': model, 'F1': new_f1, 'params': params})
        
        if new_f1 > best_f1:
            best_f1 = new_f1
            best_model = model
            best_params = params

# Sort by F1 score
results_list = sorted(results_list, key=lambda x: x['F1'], reverse=True)

# Output
print(f"Best model: {best_model}, Best F1 Score: {best_f1}")
print(f"Best hyperparameters: {best_params}")
print("All Results:")
for res in results_list:
    print(f"Model: {res['model']}, F1 Score: {res['F1']}, Params: {res['params']}")
