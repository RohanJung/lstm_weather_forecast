import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('combined_file.csv')

# Convert the 'DATE' column to datetime format
df['DATE'] = pd.to_datetime(df['DATE'])

df['PRECTOT'] = df['PRECTOT'].replace(0, float('nan')).interpolate()


# Sort the DataFrame by the 'DATE' column in ascending order
df_sorted = df.sort_values(by='DATE')

# Save the sorted DataFrame to a new CSV file


df_sorted.loc[df_sorted['DATE'] >= '2018-01-01', 'PRECTOT'] = df_sorted.loc[df_sorted['DATE'] >= '2018-01-01', 'PRECTOT'].cumsum()

# Save the sorted DataFrame to a new CSV file
df_sorted.to_csv('combined_file.csv', index=False)

# import pandas as pd
#
# # Read the first CSV file into a DataFrame
# df1 = pd.read_csv('sorted_hero.csv')
#
# # Read the second CSV file into a DataFrame
# df2 = pd.read_csv('sorted_hero2.csv')
#
# # Concatenate the two DataFrames vertically
# combined_df = pd.concat([df1, df2], ignore_index=True)
#
# # Save the combined DataFrame to a new CSV file
# combined_df.to_csv('combined_file.csv', index=False)
#
