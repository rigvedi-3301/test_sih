import pandas as pd
import os

csv_files = [
    'minitrain_data/github_1/cleansed_data.csv',
    'minitrain_data/kaggle_1/kaggle_1_clean.csv', 
    'minitrain_data/kaggle_2/kaggle_2_cleaned.csv',
    'minitrain_data/kaggle_3/kaggle_3_clean.csv'
]

dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    df['source_file'] = os.path.basename(file)  
    dataframes.append(df)
    print(f"Loaded {file} with {len(df)} rows")

combined_df = pd.concat(dataframes, ignore_index=True)

print(f"\nBefore removing duplicates: {len(combined_df)} rows")

columns_to_check = [col for col in combined_df.columns if col != 'source_file']
unique_df = combined_df.drop_duplicates(subset=columns_to_check)

print(f"After removing duplicates: {len(unique_df)} rows")
print(f"Removed {len(combined_df) - len(unique_df)} duplicate rows")

output_file = 'dataset_1.csv'
unique_df.to_csv(output_file, index=False)

print(f"\nCollated data saved to {output_file}")
print(f" Final dataset shape: {unique_df.shape}")

print("\nRows by source file:")
print(unique_df['source_file'].value_counts())