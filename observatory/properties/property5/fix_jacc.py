import os
import torch
import argparse


def update_files(root_dir):
    # recursively navigate through all subfolders and files
    for dir_name, sub_dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            # check if file ends with 'results.pt'
            if file_name.endswith('results.pt'):
                file_path = os.path.join(dir_name, file_name)
                print(file_path)
                # load file
                results_list = torch.load(file_path)
                
                # iterate over each dictionary in the list
                for results in results_list:
                    # check for 'multiset_jaccard_similarity' key and absence of 'weighted_jaccard_coeefient'
                    if 'multiset_jaccard_similarity' in results and 'weighted_jaccard_coeefient' not in results:
                        old_value = results['multiset_jaccard_similarity']
                        
                        # update keys as required
                        results['weighted_jaccard_coeefient'] = old_value
                        results['multiset_jaccard_similarity'] = old_value / (1 + old_value)

                # save the updated results back to the file
                torch.save(results_list, file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update results.pt files')
    parser.add_argument('--path', type=str, required=True, help='Path to the root directory')
    args = parser.parse_args()
    update_files(args.path)
