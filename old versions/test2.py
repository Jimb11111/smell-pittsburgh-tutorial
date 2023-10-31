def pretty_print(df, message):
    print("\n================================================")
    print("%s\n" % message)
    print(df)
    print("\nColumn names below:")
    print(list(df.columns))
    print("================================================\n")



# All your previous imports
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

from preprocessData import preprocessData
from computeFeatures import computeFeatures
from util import scorer, printScores, createSplits, computeFeatureImportance


# Hyperparameter ranges
smell_thr_values = [30, 40, 50]  # replace with the range you'd like to test
smell_predict_hrs_values = [4, 6, 8]  # replace with your range
look_back_hrs_values = [1, 2, 3]  # replace with your range
add_inter_values = [True, False]  # boolean flag, so True and False
test_size_values = [2680, 3000]  # replace with your range
train_size_values = [6360, 7000]  # replace with your range

# Models to test
models = [DummyClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier()]

best_model = None
best_f1 = 0
best_params = {}

# Loop through every combination of hyperparameters
for smell_thr in smell_thr_values:
    for smell_predict_hrs in smell_predict_hrs_values:
        for look_back_hrs in look_back_hrs_values:
            for add_inter in add_inter_values:
                for test_size in test_size_values:
                    for train_size in train_size_values:
                        # Preprocess data
                        df_sensor, df_smell = preprocessData(in_p=["dataset/esdr_raw/", "dataset/smell_raw.csv"])
                        pretty_print(df_sensor, "Display all sensor data and column names")
                        pretty_print(df_smell, "Display smell data and column names")
                        # Compute features and response
                        df_X, df_Y, _ = computeFeatures(df_esdr=df_sensor, df_smell=df_smell,
                                                        f_hr=smell_predict_hrs, b_hr=look_back_hrs,
                                                        thr=smell_thr, add_inter=False)

                        # Cross-validation splits
                        splits = createSplits(test_size, train_size, df_X.shape[0])

                        for model in models:
                            result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer)
                            new_f1 = result['test_f1'].mean()
                            if new_f1 > best_f1:
                                best_f1 = new_f1
                                best_model = model
                                best_params = {'smell_thr': smell_thr, 'smell_predict_hrs': smell_predict_hrs,
                                               'look_back_hrs': look_back_hrs, 'add_inter': add_inter,
                                               'test_size': test_size, 'train_size': train_size}

print(f"Best model: {best_model}, Best F1 Score: {best_f1}")
print(f"Best hyperparameters: {best_params}")