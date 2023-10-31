from joblib import Parallel, delayed
from preprocessData import preprocessData
from computeFeatures import computeFeatures
from util import scorer, printScores, createSplits, computeFeatureImportance
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

def pretty_print(df, message):
    print("\n================================================")
    print("%s\n" % message)
    print(df)
    print("\nColumn names below:")
    print(list(df.columns))
    print("================================================\n")

df_sensor, df_smell = preprocessData(in_p=["dataset/esdr_raw/","dataset/smell_raw.csv"])
pretty_print(df_sensor, "Display all sensor data and column names")
pretty_print(df_smell, "Display smell data and column names")

# Parallelize Feature Computation
smell_predict_hrs_list = [4, 8, 12]
look_back_hrs_list = [1, 2, 3]
smell_thr_list = [30, 40, 50]
add_inter = False

feature_results = Parallel(n_jobs=-1)(delayed(computeFeatures)(df_esdr=df_sensor, df_smell=df_smell,
        f_hr=f_hr, b_hr=b_hr, thr=thr, add_inter=add_inter) for f_hr in smell_predict_hrs_list for b_hr in look_back_hrs_list for thr in smell_thr_list)

# For demonstration, let's assume we'll use the first set of features for the next steps.
df_X, df_Y, _ = feature_results[0]
pretty_print(df_X, "Display features (X) and column names")
pretty_print(df_Y, "Display response (Y) and column names")

# Parallelize Model Evaluation
test_size = 2680
train_size = 6360
splits = createSplits(test_size, train_size, df_X.shape[0])

models_to_evaluate = [DummyClassifier(strategy="constant", constant=0), DecisionTreeClassifier(), MLPClassifier(), RandomForestClassifier()]
evaluation_results = Parallel(n_jobs=-1)(delayed(cross_validate)(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer) for model in models_to_evaluate)

# For demonstration, let's assume we'll print the results of the first model.
print("Use model", models_to_evaluate[3])
print("Perform cross-validation, please wait...")
printScores(evaluation_results[3])

feature_importance = computeFeatureImportance(df_X, df_Y, scoring="f1")
pretty_print(feature_importance, "Display feature importance based on f1-score")
