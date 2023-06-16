import torch
import matplotlib.pyplot as plt
import argparse
import os

# Set up command line argument parser
parser = argparse.ArgumentParser(description='Plotting script')
parser.add_argument('--results_file', type=str, required=True,
                    help='Path to the results.pt file')
parser.add_argument('--save_dir', type=str, required=True,
                    help='Directory to save the output picture')
parser.add_argument('--pic_name', type=str, required=True,
                    help='Name for the output picture')

args = parser.parse_args()
if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
# Load the data
results = torch.load(args.results_file)

# Unpack results into two lists
containment = [v[0] for v in results]
similarity = [v[1] for v in results]

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(similarity, containment, alpha=0.7)
plt.title('Relationship between True containment and Similarity')
plt.xlabel('Similarity')
plt.ylabel('containment')
plt.grid(True)

# Save the plot
plt.savefig(f'{args.save_dir}/{args.pic_name}.png')
