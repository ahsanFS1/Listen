import pandas as pd

df = pd.read_csv('data/psl_landmarks.csv')

print('=' * 70)
print('PSL Landmarks Label Check')
print('=' * 70)
print(f'\nTotal samples: {len(df):,}')
print(f'Total unique labels: {df["label"].nunique()}')

print('\n' + '=' * 70)
print('All Unique Labels with Sample Counts:')
print('=' * 70)

labels = sorted(df['label'].unique())
for i, label in enumerate(labels, 1):
    count = (df['label'] == label).sum()
    print(f'{i:2d}. {label:20s} - {count:5,} samples')

print('\n' + '=' * 70)

