import os
import torch
import re
import argparse

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Concatenate  .pt files.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory containing the files.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Get a list of all filenames in the directory
    filenames = os.listdir(args.path)

    # Filter out only the filenames that match the pattern 'ntom_results.pt'
    # and sort them by n to m in ascending order
    filenames = sorted(
        [
            filename
            for filename in filenames
            if re.match(r"\d+to\d+_results\.pt", filename)
        ],
        key=lambda x: (int(x.split("to")[0]), int(x.split("to")[1].split("_")[0])),
    )

    # Initialize an empty list to store all data
    all_data = []

    # Loop over the sorted filenames
    for filename in filenames:
        # Load the data from the file
        data = torch.load(os.path.join(args.path, filename))
        # Append the data to the all_data list
        all_data.extend(data)

    # Save the sorted data
    torch.save(all_data, os.path.join(args.path, "results.pt"))
