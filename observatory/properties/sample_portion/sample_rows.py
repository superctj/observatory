import pandas as pd

def sample_rows(table: pd.DataFrame, percentage: float = 1.0) -> pd.DataFrame:
    if percentage < 0 or percentage > 1:
        raise ValueError("Percentage should be between 0 and 1")
    

    # Calculate the number of rows to sample
    sample_size = int(len(table) * percentage)

    # Randomly select a subset of rows without replacement
    sampled_table = table.sample(n=sample_size, replace=False)

    return sampled_table