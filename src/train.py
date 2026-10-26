import os
import json
import joblib
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timezone
from itertools import product
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.calibration import calibration_curve


# Directories
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('charts/evaluation', exist_ok=True)


# settings
DB_HOST = 'db'
DB_PORT = 5432
DB_NAME = 'leads_db'
DB_USER = 'admin'
DB_PASSWORD = 'admin'

RANDOM_STATE = 42
AP_TOLERANCE = 0.005
K_VALUES = [500, 1000, 2000]




# Load data
with psycopg2.connect(
    host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
    user=DB_USER, password=DB_PASSWORD
) as conn:
    df = pd.read_sql_query('SELECT * FROM leads', conn)


# Data preparation
df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
df = df.sort_values('Created At').drop_duplicates('Lead ID', keep='first').copy()
df['Created Hour'] = df['Created At'].dt.hour
df['Created DayOfWeek'] = df['Created At'].dt.day_of_week

target = 'Completed Purchase'
excluded_columns = [
    'Lead ID', 'Created At', 'Completed Purchase',
    'Expected Margin', 'Incoming Call Last 24h'
]





# Define target 
X = df.drop(columns=excluded_columns)
y = df[target].astype(int)
categorical_columns = X.select_dtypes(include=['object', 'str', 'category']).columns.tolist()
numeric_columns = X.select_dtypes(include='number').columns.tolist()

print('\nDataset shape:', df.shape)
print('Categorical columns:', categorical_columns)
print('Numeric columns:', numeric_columns)
print('Purchase rate:', round(y.mean(), 4))


# Train / Validation / Test split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15 / 0.85,
    random_state=RANDOM_STATE, stratify=y_train_val
)
print('\nTrain:', X_train.shape, 'Validation:', X_val.shape, 'Test:', X_test.shape)








# Preprocessing
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

logistic_preprocessor = ColumnTransformer([
    ('numeric', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())]), numeric_columns),
    ('categorical', categorical_transformer, categorical_columns)
])

lgbm_preprocessor = ColumnTransformer([
    ('numeric', SimpleImputer(strategy='median'), numeric_columns),
    ('categorical', categorical_transformer, categorical_columns)
])


# Model configurations
logistic_configs = [
    {'C': c, 'penalty': penalty, 'class_weight': weight}
    for c, penalty, weight in product(
        [0.01, 0.1, 1],
        ['l1', 'l2'],
        [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 4}]
    )
]

lgbm_base_configs = [
    {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 15, 'min_child_samples': 50},
    {'n_estimators': 400, 'learning_rate': 0.03, 'num_leaves': 15, 'min_child_samples': 100, 'reg_lambda': 1},
    {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 7, 'max_depth': 4,
     'min_child_samples': 50, 'reg_lambda': 5}
]



positive_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
lgbm_weights = [1, 2, 5, positive_ratio]
lgbm_configs = [
    {**base, 'scale_pos_weight': weight}
    for base, weight in product(lgbm_base_configs, lgbm_weights)
]
model_configs = {
    'Logistic Regression': logistic_configs,
    'LightGBM One-Hot': lgbm_configs,
    'LightGBM Native': lgbm_configs
}


# Native categorical features
def prepare_native_data(data, categories=None):
    result = data.copy()
    for col in categorical_columns:
        if categories is None:
            result[col] = result[col].astype('category')
        else:
            result[col] = pd.Categorical(result[col], categories=categories[col])
    return result


def get_categories(data):
    return {
        col: data[col].cat.categories.tolist()
        for col in categorical_columns
    }


# Build model
def build_model(model_name, config):
    if model_name == 'Logistic Regression':
        return Pipeline([
            ('preprocessor', clone(logistic_preprocessor)),
            ('model', LogisticRegression(
                C=config['C'], l1_ratio=1.0 if config['penalty'] == 'l1' else 0.0,
                class_weight=config['class_weight'], solver='liblinear',
                max_iter=1000, random_state=RANDOM_STATE
            ))
        ])
    if model_name == 'LightGBM One-Hot':
        return Pipeline([
            ('preprocessor', clone(lgbm_preprocessor)),
            ('model', LGBMClassifier(**config, random_state=RANDOM_STATE, verbosity=-1, n_jobs=4))
        ])
    return LGBMClassifier(**config, random_state=RANDOM_STATE, verbosity=-1, n_jobs=4)


# Top-K metrics
def calculate_top_k_metrics(y_true, probabilities, k):
    actual = y_true.to_numpy()
    top_indices = pd.Series(probabilities).nlargest(min(k, len(actual))).index.to_numpy()
    purchases = int(actual[top_indices].sum())
    precision = purchases / len(top_indices)
    total_purchases = int(actual.sum())
    recall = purchases / total_purchases if total_purchases else 0.0
    baseline = actual.mean()
    lift = precision / baseline if baseline else 0.0
    return {
        'Purchases': purchases,
        'Precision': precision,
        'Recall': recall,
        'Lift': lift
    }


# Classification metrics at a fixed 0.50 threshold (descriptive, not model-selection criteria)
def classification_metrics(y_true, probabilities, threshold=0.5):
    predicted = (pd.Series(probabilities).to_numpy() >= threshold).astype(int)
    actual = y_true.to_numpy()
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        'Accuracy@0.5': accuracy_score(actual, predicted),
        'Precision@0.5': precision_score(actual, predicted, zero_division=0),
        'Recall@0.5': recall_score(actual, predicted, zero_division=0),
        'F1@0.5': f1_score(actual, predicted, zero_division=0),
        'TN@0.5': int(tn), 'FP@0.5': int(fp),
        'FN@0.5': int(fn), 'TP@0.5': int(tp)
    }


def evaluate(y_true, probabilities):
    scores = {
        'ROC_AUC': roc_auc_score(y_true, probabilities),
        'PR_AUC': average_precision_score(y_true, probabilities),
        'Brier_Score': brier_score_loss(y_true, probabilities),
        **classification_metrics(y_true, probabilities)
    }
    for k in K_VALUES:
        for name, value in calculate_top_k_metrics(y_true, probabilities, k).items():
            scores[f'{name}@{k}'] = value
    return scores


# Validation
results = []
X_train_native = prepare_native_data(X_train)
train_categories = get_categories(X_train_native)
X_val_native = prepare_native_data(X_val, train_categories)



for model_name, configs in model_configs.items():
    for config_number, config in enumerate(configs, start=1):
        model = build_model(model_name, config)
        if model_name == 'LightGBM Native':
            model.fit(X_train_native, y_train, categorical_feature=categorical_columns)
            proba = model.predict_proba(X_val_native)[:, 1]
        else:
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)[:, 1]

        metrics = evaluate(y_val, proba)
        results.append({
            'Model': model_name, 'Config': config_number,
            **metrics, 'Parameters': config
        })
        print(f'{model_name} | Config {config_number}/{len(configs)} '
              f'| AP={metrics["PR_AUC"]:.4f} '
              f'| Brier={metrics["Brier_Score"]:.4f}')


# Validation comparison
results_df = pd.DataFrame(results).sort_values('PR_AUC', ascending=False)
results_df.to_csv('reports/model_validation_comparison.csv', index=False)

best_models = results_df.drop_duplicates('Model', keep='first').copy()
best_models.to_csv('reports/best_model_configurations.csv', index=False)

# Select among near-best AP models using Brier Score
best_ap = float(results_df['PR_AUC'].max())
shortlist = results_df[results_df['PR_AUC'] >= best_ap - AP_TOLERANCE]
winner = shortlist.sort_values(['Brier_Score', 'PR_AUC'], ascending=[True, False]).iloc[0]
selected_name = winner['Model']
selected_config = winner['Parameters']
selected_config_number = int(winner['Config'])

print('\nBest validation AP:', round(best_ap, 5))
print('Selected model:', selected_name)
print('Selected config:', selected_config)
print('Selected validation AP:', round(float(winner['PR_AUC']), 5))
print('Selected validation Brier:', round(float(winner['Brier_Score']), 5))

# Evaluate the selected configuration of each family, plus the overall winner
final_candidates = best_models[['Model', 'Config']].copy()
if not ((final_candidates['Model'] == selected_name) &
        (final_candidates['Config'] == selected_config_number)).any():
    final_candidates = pd.concat([
        final_candidates,
        pd.DataFrame([{'Model': selected_name, 'Config': selected_config_number}])
    ], ignore_index=True)


# Final evaluation on Test
final_results = []
test_predictions = {}
X_train_val_native = prepare_native_data(X_train_val)
train_val_categories = get_categories(X_train_val_native)
X_test_native = prepare_native_data(X_test, train_val_categories)

for _, candidate in final_candidates.iterrows():
    model_name = candidate['Model']
    config_number = int(candidate['Config'])
    config = model_configs[model_name][config_number - 1]
    model = build_model(model_name, config)

    if model_name == 'LightGBM Native':
        model.fit(X_train_val_native, y_train_val, categorical_feature=categorical_columns)
        proba = model.predict_proba(X_test_native)[:, 1]
    else:
        model.fit(X_train_val, y_train_val)
        proba = model.predict_proba(X_test)[:, 1]

    label = f'{model_name} (Config {config_number})'
    test_predictions[label] = proba
    final_results.append({
        'Model': model_name, 'Config': config_number,
        'Selected': model_name == selected_name and config_number == selected_config_number,
        **evaluate(y_test, proba)
    })

final_df = pd.DataFrame(final_results).sort_values('PR_AUC', ascending=False)
final_df.to_csv('reports/final_model_comparison.csv', index=False)
print('\nFinal Test comparison:')
print(final_df.to_string(index=False))


# Baselines and business-focused test evaluation
baseline_rate = float(y_test.mean())
baseline_df = pd.DataFrame([{
    'Baseline': 'Always no purchase',
    'Accuracy': 1 - baseline_rate, 'Precision': 0.0,
    'Recall': 0.0, 'F1': 0.0,
    'PR_AUC_random': baseline_rate,
    'Brier_constant_prevalence': baseline_rate * (1 - baseline_rate)
}])
baseline_df.to_csv('reports/test_baselines.csv', index=False)
print('\nTest baselines:')
print(baseline_df.to_string(index=False))





# Business score uses the same urgency rules as score.py, evaluated on held-out Test
margin_test = pd.to_numeric(df.loc[X_test.index, 'Expected Margin'], errors='coerce').fillna(0).clip(lower=0).to_numpy()
expiry_test = pd.to_numeric(X_test['Days To Policy Expiry'], errors='coerce').to_numpy()
urgency_test = pd.Series(1.0, index=range(len(X_test)))
urgency_test.loc[expiry_test < 0] = 1.20
urgency_test.loc[(expiry_test >= 0) & (expiry_test <= 7)] = 1.15
urgency_test.loc[(expiry_test >= 8) & (expiry_test <= 15)] = 1.08
actual_test = y_test.to_numpy()




business_results = []
for label, proba in test_predictions.items():
    for ranking_name, ranking_scores in [
        ('Probability', proba),
        ('Business Priority', proba * margin_test * urgency_test.to_numpy())
    ]:
        order = pd.Series(ranking_scores).sort_values(ascending=False).index.to_numpy()
        for k in K_VALUES:
            idx = order[:min(k, len(order))]
            purchases = int(actual_test[idx].sum())
            rate = purchases / len(idx)
            business_results.append({
                'Model': label, 'Ranking': ranking_name, 'K': len(idx),
                'Purchases': purchases, 'Precision_at_K': rate,
                'Recall_at_K': purchases / actual_test.sum(),
                'Lift_at_K': rate / baseline_rate,
                # Proxy only: observed purchase label x expected margin, not realized profit
                'Purchased_Expected_Margin_Proxy': float((actual_test[idx] * margin_test[idx]).sum())
            })
business_df = pd.DataFrame(business_results)
business_df.to_csv('reports/test_business_evaluation.csv', index=False)
print('\nBusiness evaluation on held-out Test (expected-margin proxy, not realized profit):')
print(business_df[business_df['K'] == 1000].to_string(index=False))




# Priority segments on Test: top 20%, next 30%, bottom 50%
selected_label = f'{selected_name} (Config {selected_config_number})'
selected_business_scores = test_predictions[selected_label] * margin_test * urgency_test.to_numpy()
priority_order = pd.Series(selected_business_scores).sort_values(ascending=False).index.to_numpy()
priority_labels = pd.Series('Low', index=range(len(X_test)))
priority_labels.iloc[priority_order[:int(len(X_test) * 0.20)]] = 'High'
priority_labels.iloc[priority_order[int(len(X_test) * 0.20):int(len(X_test) * 0.50)]] = 'Medium'
priority_summary = pd.DataFrame({
    'Priority': priority_labels,
    'Purchased': actual_test,
    'Purchase_Probability': test_predictions[selected_label],
    'Expected_Margin': margin_test,
    'Purchased_Margin_Proxy': actual_test * margin_test
}).groupby('Priority', sort=False).agg(
    Leads=('Purchased', 'size'), Purchases=('Purchased', 'sum'),
    Actual_Conversion_Rate=('Purchased', 'mean'),
    Mean_Predicted_Probability=('Purchase_Probability', 'mean'),
    Mean_Expected_Margin=('Expected_Margin', 'mean'),
    Purchased_Expected_Margin_Proxy=('Purchased_Margin_Proxy', 'sum')
).reset_index()
priority_summary['Lift_vs_Test_Baseline'] = priority_summary['Actual_Conversion_Rate'] / baseline_rate
priority_summary.to_csv('reports/test_priority_segments.csv', index=False)
print('\nActual purchase rate by business priority on held-out Test:')
print(priority_summary.to_string(index=False))

# Confusion matrix of selected model at threshold 0.50
cm = confusion_matrix(actual_test, (test_predictions[selected_label] >= 0.5).astype(int), labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=13)
ax.set_xticks([0, 1], labels=['No Purchase', 'Purchase'])
ax.set_yticks([0, 1], labels=['No Purchase', 'Purchase'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Selected Model - Confusion Matrix (Threshold 0.50)')
plt.tight_layout()
plt.savefig('charts/evaluation/model_confusion_matrix.png', dpi=150)
plt.close()

# ROC curves
plt.figure(figsize=(8, 6))
for label, proba in test_predictions.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc_score(y_test, proba):.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Comparison')
plt.legend(fontsize=8)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('charts/evaluation/model_roc_curve.png', dpi=150)
plt.close()


# Precision-Recall curves
plt.figure(figsize=(8, 6))
for label, proba in test_predictions.items():
    precision, recall, _ = precision_recall_curve(y_test, proba)
    plt.plot(recall, precision, label=f'{label} (AP={average_precision_score(y_test, proba):.3f})')
plt.axhline(y_test.mean(), linestyle='--', color='gray', label=f'Baseline ({y_test.mean():.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(fontsize=8)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('charts/evaluation/model_precision_recall_curve.png', dpi=150)
plt.close()


# Lift@K
plot_k_values = [100, 250, 500, 1000, 2000, 3000, 5000]
plt.figure(figsize=(9, 6))
for label, proba in test_predictions.items():
    lifts = [calculate_top_k_metrics(y_test, proba, k)['Lift'] for k in plot_k_values]
    plt.plot(plot_k_values, lifts, marker='o', label=label)
plt.axhline(1, linestyle='--', color='gray', label='Random Baseline')
plt.xlabel('Top K Leads')
plt.ylabel('Lift@K')
plt.title('Lift by Telesales Contact Capacity')
plt.legend(fontsize=8)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('charts/evaluation/model_lift_at_k.png', dpi=150)
plt.close()


# Probability distribution
selected_label = f'{selected_name} (Config {selected_config_number})'
selected_proba = test_predictions[selected_label]
actual = y_test.to_numpy()
plt.figure(figsize=(9, 6))
plt.hist(selected_proba[actual == 0], bins=30, alpha=0.6, density=True, label='No Purchase')
plt.hist(selected_proba[actual == 1], bins=30, alpha=0.6, density=True, label='Purchase')
plt.xlabel('Predicted Purchase Probability')
plt.ylabel('Density')
plt.title(f'Probability Distribution - {selected_label}')
plt.legend()
plt.tight_layout()
plt.savefig('charts/evaluation/model_probability_distribution.png', dpi=150)
plt.close()


# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, selected_proba, n_bins=10, strategy='quantile')
plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker='o', label=selected_label)
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Observed Purchase Rate')
plt.title('Calibration Curve - Selected Model')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('charts/evaluation/model_calibration_curve.png', dpi=150)
plt.close()


# Retrain selected model on all data
selected_model = build_model(selected_name, selected_config)
native_categories = None
if selected_name == 'LightGBM Native':
    X_native = prepare_native_data(X)
    native_categories = get_categories(X_native)
    selected_model.fit(X_native, y, categorical_feature=categorical_columns)
else:
    selected_model.fit(X, y)


# Save model and metadata
trained_at = datetime.now(timezone.utc)
model_version = (
    f'{selected_name.lower().replace(" ", "_").replace("-", "_")}_'
    f'{trained_at.strftime("%Y%m%d_%H%M%S")}'
)
model_path = 'models/best_model.joblib'
metadata_path = 'reports/selected_model.json'
joblib.dump(selected_model, model_path)

metadata = {
    'model_version': model_version,
    'model_name': selected_name,
    'config_number': selected_config_number,
    'parameters': selected_config,
    'selection_metric': 'Validation AP within tolerance, then lowest Brier Score',
    'ap_tolerance': AP_TOLERANCE,
    'validation_score': float(winner['PR_AUC']),
    'validation_roc_auc': float(winner['ROC_AUC']),
    'validation_brier_score': float(winner['Brier_Score']),
    'trained_on_rows': int(len(X)),
    'feature_columns': X.columns.tolist(),
    'categorical_columns': categorical_columns,
    'native_categories': native_categories,
    'excluded_columns': excluded_columns,
    'model_path': model_path,
    'trained_at': trained_at.isoformat()
}
with open(metadata_path, 'w', encoding='utf-8') as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)

print('\nSaved model:', model_path)
print('Saved metadata:', metadata_path)
print('Model version:', model_version)
print('Charts saved in charts/evaluation/')
