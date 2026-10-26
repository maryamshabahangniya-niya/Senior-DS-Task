import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import matplotlib.pyplot as plt


# Settings
DB_HOST = 'db'
DB_PORT = 5432
DB_NAME = 'leads_db'
DB_USER = 'admin'
DB_PASSWORD = 'admin'
METADATA_PATH = 'reports/selected_model.json'

os.makedirs('charts/priority', exist_ok=True)
os.makedirs('reports', exist_ok=True)


# Load model and metadata
with open(METADATA_PATH, 'r', encoding='utf-8') as file:
    metadata = json.load(file)

model = joblib.load(metadata['model_path'])
model_name = metadata['model_name']
model_version = metadata['model_version']
feature_columns = metadata['feature_columns']

print('\nLoaded model:', model_name)
print('Model version:', model_version)


# Load leads
conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
    user=DB_USER, password=DB_PASSWORD
)

df = pd.read_sql_query('SELECT * FROM leads', conn)
df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
df = df.sort_values('Created At').drop_duplicates('Lead ID', keep='first').copy()
print('Unique leads:', len(df))
print('Feature count:', len(feature_columns))


# Feature engineering
df['Created Hour'] = df['Created At'].dt.hour
df['Created DayOfWeek'] = df['Created At'].dt.dayofweek

missing_features = sorted(set(feature_columns) - set(df.columns))
if missing_features:
    raise ValueError(f'Missing model features: {missing_features}')

X = df.loc[:, feature_columns].copy()

if model_name == 'LightGBM Native':
    categories = metadata.get('native_categories')
    if not categories:
        raise ValueError('Native LightGBM requires native_categories in metadata.')
    for col in metadata['categorical_columns']:
        X[col] = pd.Categorical(X[col], categories=categories[col])


# Predict purchase probability
df['purchase_probability'] = model.predict_proba(X)[:, 1]
if not np.isfinite(df['purchase_probability']).all():
    raise ValueError('Non-finite purchase probabilities found.')
if not df['purchase_probability'].between(0, 1).all():
    raise ValueError('Purchase probabilities are outside [0, 1].')


print('Purchase probability summary:')
print(df['purchase_probability'].describe().round(4).to_string())


# Business priority score
df['expected_margin'] = (
    pd.to_numeric(df['Expected Margin'], errors='coerce')
    .fillna(0)
    .clip(lower=0)
)

days = pd.to_numeric(df['Days To Policy Expiry'], errors='coerce')
df['urgency_weight'] = np.select(
    [days < 0, days.between(0, 7), days.between(8, 15)],
    [1.20, 1.15, 1.08],
    default=1.00
)

df['priority_score'] = (
    df['purchase_probability'] * df['expected_margin'] * df['urgency_weight']
)


print('Average expected margin:', round(df['expected_margin'].mean(), 2))
print('Urgency weights distribution:')
print(df['urgency_weight'].value_counts().sort_index().to_string())


# Assign priority
df = df.sort_values(
    ['priority_score', 'purchase_probability', 'Lead ID'],
    ascending=[False, False, True]
).copy()



n = len(df)
high_count = int(n * 0.20)
medium_count = int(n * 0.30)
df['priority'] = 'Low'
df.iloc[:high_count, df.columns.get_loc('priority')] = 'High'
df.iloc[high_count:high_count + medium_count, df.columns.get_loc('priority')] = 'Medium'



print('\nPriority distribution before saving:')
print(df['priority'].value_counts().reindex(['High', 'Medium', 'Low']).to_string())
print('Priority score summary:')
print(df['priority_score'].describe().round(2).to_string())


# Scoring metadata
scored_at = datetime.now(timezone.utc)
df['model_version'] = model_version
df['scored_at'] = scored_at


# Create scoring table and save results
rows = [
    (
        str(row['Lead ID']),
        float(row['purchase_probability']),
        float(row['expected_margin']),
        float(row['priority_score']),
        row['priority'],
        model_version,
        scored_at
    )
    for _, row in df.iterrows()
]



print('\nSaving scores to PostgreSQL (TRUNCATE + INSERT)...')
with conn.cursor() as cursor:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lead_scores (
            lead_id VARCHAR(50) PRIMARY KEY,
            purchase_probability DOUBLE PRECISION,
            expected_margin DOUBLE PRECISION,
            priority_score DOUBLE PRECISION,
            priority VARCHAR(10),
            model_version VARCHAR(100),
            scored_at TIMESTAMPTZ
        )
    ''')

    # Replace previous scoring results in the same transaction
    cursor.execute('TRUNCATE TABLE lead_scores')

    execute_values(
        cursor,
        '''
        INSERT INTO lead_scores (
            lead_id, purchase_probability, expected_margin,
            priority_score, priority, model_version, scored_at
        ) VALUES %s
        ''',
        rows,
        page_size=1000
    )

conn.commit()
print('Database transaction committed.')


# Verify saved scores in PostgreSQL
with conn.cursor() as cursor:
    cursor.execute('''
        SELECT
            COUNT(*),
            COUNT(*) FILTER (WHERE model_version = %s),
            COUNT(*) FILTER (WHERE purchase_probability IS NULL
                OR expected_margin IS NULL OR priority_score IS NULL
                OR priority IS NULL OR model_version IS NULL OR scored_at IS NULL),
            COUNT(*) FILTER (WHERE purchase_probability < 0 OR purchase_probability > 1)
        FROM lead_scores
    ''', (model_version,))
    total_rows, current_version_rows, null_rows, invalid_probability_rows = cursor.fetchone()

    cursor.execute('''
        SELECT priority, COUNT(*)
        FROM lead_scores
        WHERE model_version = %s
        GROUP BY priority
    ''', (model_version,))
    saved_priorities = dict(cursor.fetchall())

    cursor.execute('''
        SELECT lead_id, purchase_probability, expected_margin, priority_score, priority
        FROM lead_scores
        WHERE model_version = %s
        ORDER BY priority_score DESC, purchase_probability DESC, lead_id ASC
        LIMIT 10
    ''', (model_version,))
    saved_top10 = cursor.fetchall()

expected_priorities = df['priority'].value_counts().to_dict()
if total_rows != n or current_version_rows != n or null_rows or invalid_probability_rows:
    raise RuntimeError(
        f'Database verification failed: total_rows={total_rows}, current_version_rows={current_version_rows}, '
        f'expected={n}, null_rows={null_rows}, invalid_probabilities={invalid_probability_rows}'
    )
if saved_priorities != expected_priorities:
    raise RuntimeError(f'Priority counts mismatch: DB={saved_priorities}, expected={expected_priorities}')

print('\nDatabase verification passed.')
print('Total rows in lead_scores:', total_rows)
print('Rows for current model version:', current_version_rows)
print('Null rows:', null_rows)
print('Invalid probabilities:', invalid_probability_rows)
print('\nPriority distribution:', saved_priorities)
print('\nTop 10 saved leads:')
print(pd.DataFrame(
    saved_top10,
    columns=['lead_id', 'purchase_probability', 'expected_margin', 'priority_score', 'priority']
).to_string(index=False))


# Save scoring summary
summary = {
    'model_name': model_name,
    'model_version': model_version,
    'scored_at': scored_at.isoformat(),
    'unique_leads': n,
    'saved_rows_current_version': current_version_rows,
    'total_rows_in_table': total_rows,
    'null_rows': null_rows,
    'invalid_probability_rows': invalid_probability_rows,
    'priority_distribution': saved_priorities,
    'priority_rule': 'Top 20% High, next 30% Medium, remaining 50% Low',
    'score_formula': 'purchase_probability * expected_margin * urgency_weight',
    'urgency_weights': {'expired': 1.20, '0_to_7': 1.15, '8_to_15': 1.08, '16_plus_or_missing': 1.00}
}
with open('reports/scoring_summary.json', 'w', encoding='utf-8') as file:
    json.dump(summary, file, indent=4, ensure_ascii=False)


# Priority distribution chart
priority_order = ['High', 'Medium', 'Low']
counts = df['priority'].value_counts().reindex(priority_order, fill_value=0)
plt.figure(figsize=(8, 5))
counts.plot(kind='bar')
plt.title('Lead Priority Distribution')
plt.xlabel('Priority')
plt.ylabel('Number of Leads')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/priority/lead_priority_distribution.png', dpi=150)
plt.close()


# Average purchase probability by priority
probability = df.groupby('priority')['purchase_probability'].mean().reindex(priority_order)
plt.figure(figsize=(8, 5))
probability.plot(kind='bar')
plt.title('Average Purchase Probability by Priority')
plt.xlabel('Priority')
plt.ylabel('Average Purchase Probability')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/priority/priority_average_probability.png', dpi=150)
plt.close()


# Average business score by priority
business_score = df.groupby('priority')['priority_score'].mean().reindex(priority_order)
plt.figure(figsize=(8, 5))
business_score.plot(kind='bar')
plt.title('Average Business Score by Priority')
plt.xlabel('Priority')
plt.ylabel('Average Priority Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/priority/priority_average_business_score.png', dpi=150)
plt.close()

print('\nAverage purchase probability by priority:')
print(probability.round(4).to_string())
print('\nAverage business score by priority:')
print(business_score.round(2).to_string())
print('\nPriority charts saved in charts/priority/')
print('Scoring summary saved in reports/scoring_summary.json')
print('Scoring completed successfully.')


# Close database connection
conn.close()
