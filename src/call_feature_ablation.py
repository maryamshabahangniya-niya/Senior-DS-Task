import os
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss




RANDOM_STATE = 42
CALL_COLUMN = 'Incoming Call Last 24h'
os.makedirs('reports', exist_ok=True)




# Load and prepare the same lead population used in train.py
with psycopg2.connect(host='db', port=5432, dbname='leads_db', user='admin', password='admin') as conn:
    with conn.cursor() as cur:
        cur.execute('SELECT * FROM leads')
        names = [column[0] for column in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=names)




df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
df = df.sort_values('Created At').drop_duplicates('Lead ID', keep='first').copy()
df['Created Hour'] = df['Created At'].dt.hour
df['Created DayOfWeek'] = df['Created At'].dt.day_of_week



y = df['Completed Purchase'].astype(int)
base_exclusions = ['Lead ID', 'Created At', 'Completed Purchase', 'Expected Margin']
X_all = df.drop(columns=base_exclusions)



# Use identical stratified split indices in both experiments
train_val_idx, test_idx = train_test_split(
    df.index, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.15 / 0.85,
    random_state=RANDOM_STATE, stratify=y.loc[train_val_idx]
)




def top_k(y_true, probabilities, k):
    order = pd.Series(probabilities).nlargest(min(k, len(probabilities))).index.to_numpy()
    truth = y_true.to_numpy()
    purchases = int(truth[order].sum())
    precision = purchases / len(order)
    return purchases, precision, precision / truth.mean()




def fit_and_predict(features, train_rows, predict_rows):
    x_train = features.loc[train_rows]
    x_predict = features.loc[predict_rows]
    cats = x_train.select_dtypes(include=['object', 'str', 'category']).columns.tolist()
    nums = x_train.select_dtypes(include='number').columns.tolist()
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), nums),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cats)
    ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(
            C=0.01, l1_ratio=0.0, solver='liblinear',
            class_weight=None, max_iter=1000, random_state=RANDOM_STATE
        ))
    ])
    model.fit(x_train, y.loc[train_rows])
    return model.predict_proba(x_predict)[:, 1]


# A controlled feature ablation using the already-selected Logistic Regression config
rows = []
for label, include_call in [('Without Call', False), ('With Call', True)]:
    X = X_all if include_call else X_all.drop(columns=[CALL_COLUMN])
    print(f'\n{label} | features: {X.shape[1]}')

    for split_name, fit_idx, eval_idx in [
        ('Validation', train_idx, val_idx),
        ('Test', train_val_idx, test_idx)
    ]:
        probabilities = fit_and_predict(X, fit_idx, eval_idx)
        actual = y.loc[eval_idx]
        result = {
            'Experiment': label,
            'Split': split_name,
            'Features': X.shape[1],
            'ROC_AUC': roc_auc_score(actual, probabilities),
            'PR_AUC': average_precision_score(actual, probabilities),
            'Brier_Score': brier_score_loss(actual, probabilities),
        }
        for k in [500, 1000, 2000]:
            purchases, precision, lift = top_k(actual, probabilities, k)
            result[f'Purchases@{k}'] = purchases
            result[f'Precision@{k}'] = precision
            result[f'Lift@{k}'] = lift
        rows.append(result)
        print(f'{split_name}: AP={result["PR_AUC"]:.4f} | '
              f'ROC-AUC={result["ROC_AUC"]:.4f} | '
              f'Brier={result["Brier_Score"]:.4f} | '
              f'Purchases@1000={result["Purchases@1000"]}')




results = pd.DataFrame(rows)
results.to_csv('reports/call_feature_ablation.csv', index=False)
print('\nFull comparison:')
print(results.to_string(index=False))
print('\nSaved: reports/call_feature_ablation.csv')
print('Production model and scoring outputs were not changed.')
