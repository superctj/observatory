import os
import pandas as pd

def compare_directories(original_dir, modified_dir):
    # Initialize an empty dictionary to store results
    result_dict = {}

    # List the folders in the original directory
    original_folders = [d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))]

    # For each folder in the original directory
    for folder in original_folders:
        # List the corresponding folders in the modified directory
        modified_folders = [d for d in os.listdir(modified_dir) if d.startswith(folder) and os.path.isdir(os.path.join(modified_dir, d))]

        # For each corresponding folder
        for mod_folder in modified_folders:
            # List the CSV files in the original and modified folder
            original_files = [f for f in os.listdir(os.path.join(original_dir, folder)) if f.endswith('.csv')]
            modified_files = [f for f in os.listdir(os.path.join(modified_dir, mod_folder)) if f.endswith('.csv')]

            # For each CSV file in the original folder
            for file in original_files:
                # If there is a corresponding file in the modified folder
                if file in modified_files:
                    # Read the original and modified CSV files into pandas DataFrames
                    original_df = pd.read_csv(os.path.join(original_dir, folder, file))
                    modified_df = pd.read_csv(os.path.join(modified_dir, mod_folder, file))

                    # Find the columns that changed from the original table in the modified table
                    changed_columns_index = [i for i, (col_orig, col_mod) in enumerate(zip(original_df.columns, modified_df.columns)) if col_orig != col_mod]

                    # Add the result to the dictionary
                    filename_without_extension = os.path.splitext(file)[0]  # Remove .csv from file
                    key = f"{folder}_{filename_without_extension}"
                    if key not in result_dict:
                        result_dict[key] = ([original_df], [])
                    result_dict[key][0].append(modified_df)
                    result_dict[key][1].append(changed_columns_index)

    # Return the dictionary
    return result_dict




# # Call the function
# result_dict = compare_directories('processed_db_data/original', 'processed_db_data/abbreviation')
# for key, value in result_dict.items():
#     print(f"\n\nFor key {key}, the tables are:")
#     for df in value[0]:  # Iterate over each DataFrame in the list
#         print(df)  # To display DataFrame. Be careful, it could be big.
#         break
#     print(f"The columns that changed from original table in each modified table are: {value[1]}")
