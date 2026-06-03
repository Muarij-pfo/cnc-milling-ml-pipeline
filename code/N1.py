import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import *

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from scipy.stats import skew, kurtosis, zscore, ttest_rel
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import make_scorer
import random
random.seed(42)
np.random.seed(42)


df = pd.read_excel("experiment_03..xlsx")

# Target 2 creation.

#Feature Engineering (01)
df['Resultant_Velocity'] = np.sqrt(
    df['X1_ActualVelocity']**2 +
    df['Y1_ActualVelocity']**2 +
    df['Z1_ActualVelocity']**2)
    
df['Machine_State'] = pd.qcut(
    df['Resultant_Velocity'],
    3,
    labels=['Low_Speed', 'Medium_Speed', 'High_Speed']
)


#TARGETS
target_reg = 'S1_OutputPower'
target_clf1 = 'Machining_Process'
target_clf2 = 'Machine_State'


X = df.drop([target_reg, target_clf1, target_clf2], axis=1)



print("\n DESCRIPTIVE STATISTICS")
stats = df.describe().T
stats['skew'] = df.skew(numeric_only=True)
stats['kurtosis'] = df.kurtosis(numeric_only=True)
print(stats)

df.hist(figsize=(15,12))
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
plt.show()

#CLeaning
df.fillna(df.mean(numeric_only=True), inplace=True)


# OUTLIERS REMOVAL
num_df = df.select_dtypes(include=np.number)

# Removing constant columns
num_df = num_df.loc[:, num_df.std() != 0]

# Computing z-score
z_scores = np.abs(zscore(num_df, nan_policy='omit'))

# Keeping rows where at least 95% features are valid
df = df[(z_scores < 3).mean(axis=1) > 0.95]

# Reseting index
df.reset_index(drop=True, inplace=True)

print("Remaining samples after outlier removal:", len(df))






# FEATURE ENGINEERING (02)
df['Spindle_Power_Calc'] = df['S1_OutputCurrent'] * df['S1_OutputVoltage']



# Encoding
le1 = LabelEncoder()
le2 = LabelEncoder()

df[target_clf1] = le1.fit_transform(df[target_clf1])
df[target_clf2] = le2.fit_transform(df[target_clf2])



# Target selections
X = df.drop([target_reg, target_clf1, target_clf2], axis=1)

y_clf = df[[target_clf1, target_clf2]]   
y_reg = df[[target_reg]]                

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



#Splitting Dataset

X_train, X_temp, yclf_train, yclf_temp, yreg_train, yreg_temp = train_test_split(
    X_scaled, y_clf, y_reg,
    test_size=0.30,
    random_state=42,
    stratify=y_clf[target_clf1]  
)

X_val, X_test, yclf_val, yclf_test, yreg_val, yreg_test = train_test_split(
    X_temp, yclf_temp, yreg_temp,
    test_size=0.50,
    random_state=42
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


def multi_output_accuracy(y_true, y_pred):
    if hasattr(y_true, "values"):
        y_true = y_true.values
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values

    acc1 = accuracy_score(y_true[:, 0], y_pred[:, 0])
    acc2 = accuracy_score(y_true[:, 1], y_pred[:, 1])

    return (acc1 + acc2) / 2
multi_scorer = make_scorer(multi_output_accuracy)




# HYPERPARAMETER TUNING
from sklearn.model_selection import GridSearchCV

def tune_models(X, y):

    tuned_models = {}

    # Logistic Regression
    log_model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    log_params = {
        "estimator__C": [0.1, 1, 10]
    }

    grid_log = GridSearchCV(log_model, log_params, cv=5, scoring=multi_scorer, n_jobs=1)
    grid_log.fit(X, y)
    tuned_models["Logistic"] = grid_log.best_estimator_

    print("Best Logistic Params:", grid_log.best_params_)

    # SVM
    svm_model = MultiOutputClassifier(SVC(probability=True))
    svm_params = {
        "estimator__C": [0.1, 1, 10],
        "estimator__kernel": ['linear', 'rbf']
    }

    grid_svm = GridSearchCV(svm_model, svm_params, cv=5, scoring=multi_scorer, n_jobs=1)
    grid_svm.fit(X, y)
    tuned_models["SVM"] = grid_svm.best_estimator_

    print("Best SVM Params:", grid_svm.best_params_)

    # Random Forest
    rf_model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    rf_params = {
        "estimator__n_estimators": [100, 200],
        "estimator__max_depth": [5, 10, None]
    }

    grid_rf = GridSearchCV(rf_model, rf_params, cv=5, scoring=multi_scorer, n_jobs=1)
    grid_rf.fit(X, y)
    tuned_models["RandomForest"] = grid_rf.best_estimator_

    print("Best RF Params:", grid_rf.best_params_)

    return tuned_models




# MODEL TRAINING 

def train_models(X, y, model_type='clf'):

    if model_type == 'clf':
        models = {
            "Logistic": MultiOutputClassifier(
                LogisticRegression(max_iter=1000, class_weight='balanced')
            ),

            "SVM": MultiOutputClassifier(
                SVC(probability=True, class_weight='balanced')
            ),

            "RandomForest": MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    class_weight='balanced',
                    random_state=42
                )
            )
        }

    else:
        models = {
            "Linear": MultiOutputRegressor(
                LinearRegression()
            ),

            "RandomForest": MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42
                )
            ),

            "SVR": MultiOutputRegressor(
                SVR()
            )
        }

    trained = {}

    for name, model in models.items():
        model.fit(X, y)
        trained[name] = model

    return trained


clf_models = tune_models(X_train, yclf_train)
reg_models = train_models(X_train, yreg_train, 'reg')






#TABLES ETC.
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("\n===== FULL CLASSIFICATION REPORT =====")

for name, model in clf_models.items():
    pred = model.predict(X_test)

    print(f"\n🔹 Model: {name}")

    # Target 1
    print("\n--- Target 1: Machining Process ---")
    print(classification_report(yclf_test.iloc[:,0], pred[:,0],zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(yclf_test.iloc[:,0], pred[:,0]))

    # Target 2
    print("\n--- Target 2: Machine_State Condition ---")
    print(classification_report(yclf_test.iloc[:,1], pred[:,1],zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(yclf_test.iloc[:,1], pred[:,1]))




    print("\n===== FULL REGRESSION METRICS =====")

def regression_metrics(model, X, y):
    pred = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)

    n, p = X.shape
    adj_r2 = 1 - (1 - r2)*(n-1)/(n-p-1)

    return rmse, mae, r2, adj_r2

for name, model in reg_models.items():
    rmse, mae, r2, adj_r2 = regression_metrics(model, X_test, yreg_test)
    print(f"\n{name}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"Adjusted R2: {adj_r2}")


    print("\n===== MODEL COMPARISON TABLE =====")

comparison = []

for name, model in clf_models.items():
    pred = model.predict(X_test)

    acc1 = accuracy_score(yclf_test.iloc[:,0], pred[:,0])
    acc2 = accuracy_score(yclf_test.iloc[:,1], pred[:,1])

    comparison.append([name, acc1, acc2])

df_comp = pd.DataFrame(comparison, columns=["Model", "Accuracy_Target1", "Accuracy_Target2"])

print(df_comp)



# =========================================================
# 13. EVALUATION
# =========================================================
def eval_clf(model, X, y):
    pred = model.predict(X)
    return [
        accuracy_score(y.iloc[:,0], pred[:,0]),
        accuracy_score(y.iloc[:,1], pred[:,1])
    ]

def eval_reg(model, X, y):
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    return [rmse, r2]

# =========================================================
# 14. RESULTS
# =========================================================
print("\n=== CLASSIFICATION RESULTS ===")
for name, model in clf_models.items():
    print(name, eval_clf(model, X_test, yclf_test))

print("\n=== REGRESSION RESULTS ===")
for name, model in reg_models.items():
    print(name, eval_reg(model, X_test, yreg_test))



# =========================================================
# FEATURE IMPORTANCE / COEFFICIENTS
# =========================================================

feature_names = X.columns

for name, model in clf_models.items():

    print(f"\n===== FEATURE ANALYSIS: {name} =====")

    try:
        # Random Forest
        if "RandomForest" in name:
            importances = model.estimators_[0].feature_importances_
            for f, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]:
                print(f"{f}: {imp:.4f}")

        # Logistic Regression
        elif "Logistic" in name:
            coefs = model.estimators_[0].coef_[0]
            for f, c in sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1]))[:10]:
                print(f"{f}: {c:.4f}")

        # SVM (only if linear kernel)
        elif "SVM" in name and hasattr(model.estimators_[0], "coef_"):
            coefs = model.estimators_[0].coef_[0]
            for f, c in sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1]))[:10]:
                print(f"{f}: {c:.4f}")

        else:
            print("No direct feature importance available")

    except Exception as e:
        print("Error extracting importance:", e)

# =========================================================
# 15. RESIDUAL PLOT
# =========================================================
best_name = min(
    reg_models,
    key=lambda k: regression_metrics(reg_models[k], X_test, yreg_test)[0]
)

best_reg = reg_models[best_name]

print("Best Regression Model:", best_name)
residuals = yreg_test.values - best_reg.predict(X_test)

y_pred = best_reg.predict(X_test)

plt.figure()
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red')
plt.xlabel("Predicted Output Power")
plt.ylabel("Residuals")
plt.title("Residual Plot (Improved)")
plt.grid()
plt.show()

plt.figure()
plt.scatter(yreg_test, y_pred)
plt.xlabel("Actual Output Power")
plt.ylabel("Predicted Output Power")
plt.title("Actual vs Predicted")
plt.grid()
plt.show()

# =========================================================
# 16. LEARNING CURVE
# =========================================================



def plot_learning_curve_single(model, name, X, y):

    scorer = make_scorer(multi_output_accuracy)

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y.values,
        cv=5,
        scoring=scorer,
        train_sizes=np.linspace(0.2, 1.0, 5)
    )

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label="Train")
    plt.plot(train_sizes, test_mean, 'o-', label="Validation")

    plt.title(f"{name} Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

plot_learning_curve_single(clf_models["Logistic"], "Logistic", X_train, yclf_train)
plot_learning_curve_single(clf_models["SVM"], "SVM", X_train, yclf_train)
plot_learning_curve_single(clf_models["RandomForest"], "RandomForest", X_train, yclf_train)

# =========================================================
# 17. STATISTICAL TEST
# =========================================================


def compare_models_statistically(model1, model2, X, y):

    scorer = make_scorer(multi_output_accuracy)

    scores1 = cross_val_score(model1, X, y.values, cv=10, scoring=scorer)
    scores2 = cross_val_score(model2, X, y.values, cv=10, scoring=scorer)

    print("\n Model RF CV Scores:", scores1)
    print("\nModel SVM CV Scores:", scores2)

    t_stat, p_val_t = ttest_rel(scores1, scores2)
    w_stat, p_val_w = wilcoxon(scores1, scores2)

    print("\nPaired t-test p-value:", p_val_t)
    print("Wilcoxon p-value:", p_val_w)

    if p_val_w < 0.05:
        print("\n✅ Statistically significant difference between RF and SVM (RF better in reality) ")
    else:
        print("\n❌ No statistically significant difference between RF and SVM (RF better by chance)")

compare_models_statistically(
    clf_models["RandomForest"],
    clf_models["SVM"],
    X_train,
    yclf_train
)

def compare_models_statistically(model1, model3, X, y):

    scorer = make_scorer(multi_output_accuracy)

    scores1 = cross_val_score(model1, X, y.values, cv=10, scoring=scorer)
    scores3 = cross_val_score(model3, X, y.values, cv=10, scoring=scorer)

    print("\n Model RF CV Scores:", scores1)
    print("\nModel Logistic CV Scores:", scores3)

    t_stat, p_val_t = ttest_rel(scores1, scores3)
    w_stat, p_val_w = wilcoxon(scores1, scores3)

    print("\nPaired t-test p-value:", p_val_t)
    print("Wilcoxon p-value:", p_val_w)

    if p_val_w < 0.05:
        print("\n✅ Statistically significant difference between RF and Logistic (RF better in reality) ")
    else:
        print("\n❌ No statistically significant difference between RF and Logistic (RF better by chance)")

compare_models_statistically(
    clf_models["RandomForest"],
    clf_models["Logistic"],
    X_train,
    yclf_train
)





rmse, mae, r2, adj_r2 = regression_metrics(best_reg, X_test, yreg_test)

y_range = yreg_test.max() - yreg_test.min()
relative_rmse = rmse / y_range.values[0]

print("\n===== ENGINEERING INTERPRETATION =====")
print(f"RMSE: {rmse}")
print(f"Relative RMSE: {relative_rmse*100:.2f}%")

if relative_rmse < 0.1:
    print(" Excellent model (very accurate for engineering use)")
elif relative_rmse < 0.2:
    print("Acceptable but may need improvement")
else:
    print(" Not reliable for engineering application")
print("\n===== THE END =====")




