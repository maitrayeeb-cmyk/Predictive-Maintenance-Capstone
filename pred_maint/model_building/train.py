
# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from imblearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score, ConfusionMatrixDisplay
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import matplotlib.pyplot as plt
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

# Add logs to mlflow artifacts
epoch_num=1;
import logging
logging.basicConfig(filename="run.log", level=logging.INFO)
logging.info(f"epoch {epoch_num}:")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("PredictiveMaintenanceCapstone_XGB_Train_experiment1")

api = HfApi()


Xtrain_path = "hf://datasets/mrhea/pred-maint/Xtrain.csv"
Xtest_path = "hf://datasets/mrhea/pred-maint/Xtest.csv"
ytrain_path = "hf://datasets/mrhea/pred-maint/ytrain.csv"
ytest_path = "hf://datasets/mrhea/pred-maint/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

target='Engine Condition'

# List of numerical features in the dataset
numeric_features = Xtrain.columns.tolist()
print(f"Numeric features:'{numeric_features}'")


# List of categorical features in the dataset
categorical_features = []

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features) #Retaining the spec on OneHotEncoder for my learning, it is not necessary here, since there are no categorical features.
)

#-----------------------------------------------------
# No more manually computing scale_pos_weight from resampled counts —
# SMOTE is what handles the imbalance now, so leave scale_pos_weight at its default (1).
xgb_model = xgb.XGBClassifier(random_state=42)

# Order matters: scale first, resample second, classify last.
# SMOTE uses nearest-neighbor distances, so it should see scaled features, not raw ones.
model_pipeline = make_pipeline(
    preprocessor,
    SMOTE(random_state=42),
    xgb_model
)


param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1, 0.15],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1,scoring=scoring, refit='f1') # <-- changed from 'recall': pure recall is gameable by an
                                                                                                        #     always-predict-positive model when that class is the
                                                                                                        #     majority; f1 forces a precision/recall trade-off.


    grid_search.fit(Xtrain, ytrain) # original+feature-engineered, imbalanced data —
                                    # SMOTE inside the pipeline resamples each
                                    # training fold on its own; the held-out
                                    # fold in every CV split stays real, unresampled data.

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_accuracy = results['mean_test_accuracy'][i]
        std_accuracy = results['std_test_accuracy'][i]
        mean_precision = results['mean_test_precision'][i]
        mean_recall = results['mean_test_recall'][i]
        std_recall = results['std_test_recall'][i]
        mean_f1 = results['mean_test_f1'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_accuracy", mean_accuracy)
            mlflow.log_metric("std_test_accuracy", std_accuracy)
            mlflow.log_metric("mean_test_precision", mean_precision)
            mlflow.log_metric("mean_test_recall", mean_recall)
            mlflow.log_metric("std_test_recall", std_recall)
            mlflow.log_metric("mean_test_f1", mean_f1)


    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_
    print(f"best model:{best_model.get_params()}")

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]   # SMOTE step is skipped automatically here
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    print(train_report)
    print(test_report)

    
    print(f"classification report:{classification_report(ytest, y_pred_test)}")

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Plot and log confusion matrix for train set
    cm_display_train = ConfusionMatrixDisplay.from_predictions(ytrain, y_pred_train, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Train Set')
    plt.savefig('confusion_matrix_train.png')
    mlflow.log_artifact('confusion_matrix_train.png', artifact_path="plots")
    plt.close()

    # Plot and log confusion matrix for test set
    cm_display_test = ConfusionMatrixDisplay.from_predictions(ytest, y_pred_test, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Test Set')
    plt.savefig('confusion_matrix_test.png')
    mlflow.log_artifact('confusion_matrix_test.png', artifact_path="plots")
    plt.close()

    # Save the model locally
    model_path = "best_pred_maint_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    #mlflow.log_text(f"Model saved as artifact at: {model_path}",artifact_file="log.txt")

    # Extract and log feature importances
    xgb_final_model = best_model.named_steps['xgbclassifier']
    feature_importances = xgb_final_model.feature_importances_

    # Get feature names after preprocessing
    numerical_feature_names = numeric_features
    # Ensure one_hot_encoder is only accessed if categorical_features is not empty
    if categorical_features:
        one_hot_encoder = best_model.named_steps['columntransformer'].named_transformers_['onehotencoder']
        categorical_feature_names_ohe = one_hot_encoder.get_feature_names_out(categorical_features)
    else:
        categorical_feature_names_ohe = []

    all_feature_names = numerical_feature_names + list(categorical_feature_names_ohe)

    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(feature_importance_df)
    mlflow.log_dict(feature_importance_df.set_index('Feature').to_dict()['Importance'], "feature_importances")

    #Log the print statements
    mlflow.log_artifact("run.log")

    # Upload to Hugging Face space/model
    repo_id = "mrhea/pred-maint"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        # print(f"Space '{repo_id}' already exists. Using it.")
        #mlflow.log_text(f"Space '{repo_id}' already exists. Using it.", artifact_path="logs")
    except RepositoryNotFoundError:
      #No existing repo with same name, creating one.
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")
        mlflow.log_text(f"Space '{repo_id}' created.", artifact_file="log.txt")


    # Step 2: Upload the best model to HF pred-maint space/model
    api.upload_file(
        path_or_fileobj="best_pred_maint_v1.joblib",
        path_in_repo="best_pred_maint_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )




