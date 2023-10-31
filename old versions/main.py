"""

The following is a brief description of the pipeline:
- Step 1: Preprocess the raw data
- Step 2: Select variables from the preprocessed sensor data
- Step 3: Extract features (X) and the response (Y) from the preprocessed data
- Step 4: Train and evaluate a machine learning model (F) that maps X to Y
- Step 5: Investigate the importance of each feature
"""


# This is a reusable function to print the data
# (no need to modify this part)
def pretty_print(df, message):
    print("\n================================================")
    print("%s\n" % message)
    print(df)
    print("\nColumn names below:")
    print(list(df.columns))
    print("================================================\n")


# Import the "preprocessData" function in the "preprocessData.py" script for reuse
# (no need to modify this part)
from preprocessData import preprocessData

# Preprocess and print sensor and smell data
# (no need to modify this part)
df_sensor, df_smell = preprocessData(in_p=["dataset/esdr_raw/","dataset/smell_raw.csv"])
pretty_print(df_sensor, "Display all sensor data and column names")
pretty_print(df_smell, "Display smell data and column names")


# Select some variables, which means the columns in the data table.
# (you may want to modify this part to add more variables for experiments)
# (you can also comment out the following two lines to indicate that you want all variables)
# wanted_cols = ["DateTime", "3.feed_24.PM10_UG_M3", "3.feed_27.CO_PPB", "3.feed_28.SONICWS_MPH", "3.feed_26.SONICWS_MPH"]
# df_sensor = df_sensor[wanted_cols]

# Print the selected sensor data
# (no need to modify this part)
pretty_print(df_sensor, "Display selected sensor data and column names")


# Import the "computeFeatures" function in the "computeFeatures.py" script for reuse
# (no need to modify this part)
from computeFeatures import computeFeatures

# Indicate the threshold to define a smell event
# (you may want to modify this parameter for experiments)
smell_thr = 40

# Indicate the number of future hours to predict smell events
# (you may want to modify this parameter for experiments)
smell_predict_hrs = 8

# Indicate the number of hours to look back to check previous sensor data
# (you may want to modify this parameter for experiments)
look_back_hrs = 1

# Indicate if you want to add interaction terms in the features (like x1*x2)
# (you may want to modify this parameter for experiments)
add_inter = False

# Compute and print features (X) and response (Y)
# (no need to modify this part)
df_X, df_Y, _ = computeFeatures(df_esdr=df_sensor, df_smell=df_smell,
        f_hr=smell_predict_hrs, b_hr=look_back_hrs, thr=smell_thr, add_inter=add_inter)
pretty_print(df_X, "Display features (X) and column names")
pretty_print(df_Y, "Display response (Y) and column names")


"""

Parameter "test_size" is the number of samples for testing.
For example, setting it to 168 means using 168 samples for testing
, which also means 168 hours (or 7 days) of data.

Parameter "train_size" is the number of samples for training.
For example, setting it to 336 means using 336 samples for testing
, which also means 336 hours (or 14 days) of data.

"""

# Import packages for reuse
# (you may want to import more models)
from util import scorer
from util import printScores
from util import createSplits

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_validate

# Indicate how much data you want to use to test the model
# (you may want to modify this parameter for experiments)
test_size = 2680

# Indicate how much data you want to use to train the model
# (you may want to modify this parameter for experiments)
train_size = 6360

# Build the cross validation splits
# (no need to modify this part)
splits = createSplits(test_size, train_size, df_X.shape[0])

# Indicate which model you want to use to predict smell events
# (you may want to modify this part to use other models)

# model = DummyClassifier(strategy="constant", constant=0)
# model = DecisionTreeClassifier()
model = MLPClassifier()
# model = RandomForestClassifier()

# Perform cross-validation to evaluate the model
# (no need to modify this part)
print("Use model", model)
print("Perform cross-validation, please wait...")
result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer)
printScores(result)

# Import packages for reuse
# (no need to modify this part)
from util import computeFeatureImportance

# Compute and show feature importance weights
# (no need to modify this part)
feature_importance = computeFeatureImportance(df_X, df_Y, scoring="f1")
pretty_print(feature_importance, "Display feature importance based on f1-score")
