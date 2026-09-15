
from imblearn.over_sampling import SMOTE
# for data manipulation
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/mrhea/pred-maint/engine_data.csv"

# STEP 3: Loading the data from the HF-space to local
engine_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Assigning engine_dataset to data for consistency with previous analysis
data = engine_dataset.copy()

# Define the target variable for the classification task
target='Engine Condition'

# List of numerical features in the dataset
numeric_features = [
    'Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure','lub oil temp', 'Coolant temp'
]


# STEP 4: Data Pre-processing
# Checking for duplicate records in data
if data.duplicated().sum()>0:
    print(f"Duplicate records found in the dataset. Number of duplicate records: {data.duplicated().sum()}")
    # Dropping the duplicate records from data
    data.drop_duplicates(inplace=True)
    print("Duplicate records dropped successfully.")
else:
    print("No duplicate records found in the dataset.")

# Checking for null records in data
null_records=data.isnull().sum()
for col,null_records_count in null_records.items():
    print(f"Null records found in column {col}: {null_records_count}")
    if null_records_count>0:
        print(f"Null records found in the dataset. Number of null records: {null_records}")
        # Imputing median values in the records with null values
        print("Imputing null records with median values...")
        imputer=SimpleImputer(strategy='median')
        data[col]=imputer.fit_transform(data[[col]])
        print("Null records imputed successfully.")
    else:
        print("No null records found in the dataset.")

print(f"data.head(): {data.head()}")
print(f"data.info(): {data.info()}")
###################################################
#Improving the model

# Adding the five calculated parameters
data['Specific Lubrication Index'] = data['Lub oil pressure'] / data['Engine rpm']
data['Volumetric Coolant Flow Proxy'] = data['Coolant pressure'] / data['Engine rpm']
data['Thermal Crossover Delta'] = data['lub oil temp'] - data['Coolant temp']
data['Safety Metric'] = data['Coolant pressure'] / data['Coolant temp']
data['Oil-to-Coolant Pressure Differential'] = data['Lub oil pressure'] - data['Coolant pressure']

#dropping the Coolant columns to reduce multi-collinearity
data=data.drop(['Coolant pressure','Coolant temp'],axis=1)
print(f"data.head(): {data.head()}")
print(f"data.info(): {data.info()}")
numeric_features = data.columns.drop('Engine Condition')
print(numeric_features)


##################################################

# Define predictor matrix (X) using the numeric features
X = data[numeric_features]

# Define target variable
y = data[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,    # Ensures reproducibility by setting a fixed random seed
    stratify=y         # Ensures train/test splits are proportional
)

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"class_weight={class_weight}")


print("Completed train-test split")
Xtrain.to_csv("pred_maint/data/Xtrain.csv",index=False)
Xtest.to_csv("pred_maint/data/Xtest.csv",index=False)
ytrain.to_csv("pred_maint/data/ytrain.csv",index=False)
ytest.to_csv("pred_maint/data/ytest.csv",index=False)

print("Completed writing train and test data files")
files = [
    "pred_maint/data/Xtrain.csv",
    "pred_maint/data/Xtest.csv",
    "pred_maint/data/ytrain.csv",
    "pred_maint/data/ytest.csv",
]

# STEP 5: Uploading Train and Test dataset files to HF space<repo_id>/dataset<repo_type>
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="mrhea/pred-maint",
        repo_type="dataset",
    )
