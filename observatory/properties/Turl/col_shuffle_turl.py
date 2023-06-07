import torch
import torch.nn as nn
import os

from observatory.models.TURL.model.configuration import TableConfig
from observatory.models.TURL.model.model import HybridTableMaskedLM
from analyze_embeddings import analyze_embeddings
import time


def set_timer(flag_list):
    time.sleep(600)  # 600 seconds == 10 minutes
    flag_list[0] = True

def load_turl_model(config, ckpt_path: str):
    model = HybridTableMaskedLM(config, is_simple=True)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    return model


class TURL(nn.Module):
    def __init__(self, config_name: str, ckpt_path: str):
        super(TURL, self).__init__()

        config = TableConfig.from_pretrained(config_name)
        self.model = load_turl_model(config, ckpt_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_column_embeddings(self, input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent_text, input_ent_text_length, input_ent, input_ent_type, input_ent_mask):
        tok_outputs, ent_outputs, _ = self.model.table(input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent_text, input_ent_text_length, None, input_ent, input_ent_type, input_ent_mask, None)
            
        tok_sequence_output = self.dropout(tok_outputs[0])
        tok_col_output = torch.matmul(column_header_mask, tok_sequence_output)
        tok_col_output /= column_header_mask.sum(dim=-1,keepdim=True).clamp(1.0,9999.0)

        ent_sequence_output = self.dropout(ent_outputs[0])
        ent_col_output = torch.matmul(column_entity_mask, ent_sequence_output)
        ent_col_output /= column_entity_mask.sum(dim=-1,keepdim=True).clamp(1.0,9999.0)

        col_embeddings = torch.cat([tok_col_output, ent_col_output], dim=-1)
        return col_embeddings
    

def reorder_embeddings(embeddings1, embeddings2):
    # Expand dimensions for cosine similarity calculation
    embeddings1_exp = embeddings1.unsqueeze(1)
    embeddings2_exp = embeddings2.unsqueeze(0)

    # Compute cosine similarity
    cos_sim_matrix = torch.nn.functional.cosine_similarity(embeddings1_exp, embeddings2_exp, dim=2)

    # Get the indices of max similarity
    _, indices_max_sim = torch.max(cos_sim_matrix, dim=1)
    print(indices_max_sim)

    # Create an empty tensor of the same size on the same device to store the reordered embeddings
    reordered_embeddings2 = torch.empty_like(embeddings2)

    # Reorder the second embeddings according to the indices of max similarity
    for i, index in enumerate(indices_max_sim):
        reordered_embeddings2[i] = embeddings2[index]

    return reordered_embeddings2



if __name__ == "__main__":
    import argparse

    from observatory.models.TURL.data_loader.CT_Wiki_data_loaders import CTLoader
    from observatory.models.TURL.model.transformers import BertTokenizer
    from observatory.models.TURL.utils.util import load_entity_vocab

    from col_shuffle_turl_wiki_tables import TurlWikiTableDataset

    parser = argparse.ArgumentParser(description='Process tables and save embeddings.')
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='Directory that contains TURL specific files such as entity vocabulary')
    parser.add_argument('--config_path', type=str, required=True, help='Path to TURL model config')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to TURL model checkpoint')
    parser.add_argument('--cuda_device', type=int, default=None, help='Path to TURL model checkpoint')
    parser.add_argument('-s', '--save_directory', type=str, required=True, help='Directory to save embeddings to')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('-l', '--start_line', type=int, default=0, help='The index of start table')

    args = parser.parse_args()
    
    min_ent_count = 2
    entity_vocab = load_entity_vocab(args.data_dir, ignore_bad_title=True, min_ent_count=min_ent_count)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if args.cuda_device:
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        device = torch.device("cpu")
    
    model = TURL(args.config_path, args.ckpt_path)
    model.to(device)
     # save_directory_results  = os.path.join('/nfs/turbo/coe-jag/zjsun', 'sample_portion', str(args.sample_portion), args.save_directory, model_name ,'results')
    # save_directory_embeddings  = os.path.join('/nfs/turbo/coe-jag/zjsun','sample_portion', str(args.sample_portion), args.save_directory, model_name ,'embeddings')
    # save_directory_results  = os.path.join('/nfs/turbo/coe-jag/zjsun', 'col_insig', args.save_directory, 'Turl' ,'results')
    # save_directory_embeddings  = os.path.join('/nfs/turbo/coe-jag/zjsun', 'col_insig', args.save_directory, 'Turl' ,'embeddings')
    # save_directory_results  = os.path.join( 'col_insig', args.save_directory, 'Turl' ,'results')
    # save_directory_embeddings  = os.path.join( 'col_insig', args.save_directory, 'Turl' ,'embeddings')
    save_directory_results  = os.path.join("/ssd/congtj/observatory/experiments", 'col_insig', args.save_directory, 'Turl' ,'results')
    save_directory_embeddings  = os.path.join("/ssd/congtj/observatory/experiments", 'col_insig', args.save_directory, 'Turl' ,'embeddings')
    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
    
    with open(os.path.join(args.data_dir, "test_tables.jsonl"), "r") as f:
        lines = f.readlines()
        for table_index in range(args.start_line, len(lines)):
            line = lines[table_index]
            
            # timer_flag = [False]
            # timer_thread = threading.Thread(target=set_timer, args=(timer_flag,))
            # timer_thread.start()

            try:
                test_dataset = TurlWikiTableDataset(line, entity_vocab, tokenizer, split="test", force_new=False)
                perms = test_dataset.perms
                keepeded_entity_columns = test_dataset.keepeded_entity_columns
            except Exception as e:
                print(f"table{table_index} failed.")
                try:
                    print(e)
                except:
                    print()
                print(line)
                continue  # if an error occurs, continue to the next line

            # while timer_thread.is_alive():
            #     time.sleep(1)  # check every second

            # if timer_flag[0]:
            #     print("Operation took too long, moving to next iteration.")
                # continue
            # test_dataset = TurlWikiTableDataset(line, entity_vocab, tokenizer, split="test", force_new=False)
            # perms = test_dataset.perms
            # keepeded_entity_columns = test_dataset.keepeded_entity_columns    
                
                
                
            if len(test_dataset) < 24:
                continue
            
            
            
            test_dataloader = CTLoader(test_dataset, batch_size=args.batch_size, shuffle=False, is_train=False)
            all_embeddings = []
            original_entity_column =  keepeded_entity_columns[0]  

            for batch in test_dataloader:
                table_ids, input_tok, input_tok_type, input_tok_pos, input_tok_mask, \
                    input_ent_text, input_ent_text_length, input_ent, input_ent_type, input_ent_mask, column_entity_mask, column_header_mask, labels_mask, labels = batch
                input_tok = input_tok.to(device)
                input_tok_type = input_tok_type.to(device)
                input_tok_pos = input_tok_pos.to(device)
                input_tok_mask = input_tok_mask.to(device)
                input_ent_text = input_ent_text.to(device)
                input_ent_text_length = input_ent_text_length.to(device)
                input_ent = input_ent.to(device)
                input_ent_type = input_ent_type.to(device)
                input_ent_mask = input_ent_mask.to(device)
                column_entity_mask = column_entity_mask.to(device)
                column_header_mask = column_header_mask.to(device)
                labels_mask = labels_mask.to(device)
                labels = labels.to(device)
                
                with torch.no_grad():
                    col_embeddings = model.get_column_embeddings(input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent_text, input_ent_text_length, input_ent, input_ent_type, input_ent_mask)
                    # col_embeddings = col_embeddings.squeeze(0)

                    # print("=" * 50)
                    # print("Number of columns: ", labels.shape)
                    # print("Column embeddings shape: ", col_embeddings.shape)
                    # print(col_embeddings.size())
                    if all_embeddings == []:
                        all_embeddings = col_embeddings
                        # print(all_shuffled_embeddings.size())
                    else:
                        # print(all_shuffled_embeddings.size())
                        if len(original_entity_column) == len(col_embeddings[0]):
                            all_embeddings = torch.cat((all_embeddings, col_embeddings), dim=0)
                        else:
                            print(f"{len(col_embeddings[0])} is not equal to {len(original_entity_column)}")
                            print("\n\n\n This shouldn't occur !!!!\n\n\n")
            ##############introduce new machneism to know which column was killed        
            print(len(all_embeddings))
            print()            
            print(original_entity_column)
            print()
            all_shuffled_embeddings =[all_embeddings[0]]
            for j in range(1, len(perms)):
                ordered_embeddings = reorder_embeddings(all_embeddings[0], all_embeddings[j])

                perm = perms[j]
                entity_column = keepeded_entity_columns[j]
                new_perm = [entity_column.index(index) for index in original_entity_column]
                # new_perm = [original_entity_column.index(index) for index in entity_column]

                # print(entity_column)
                print(new_perm)
                # Create a list of the same length as perm, filled with None
                # ordered_embeddings = reorder_embeddings(all_embeddings[0], all_embeddings[j])
                
                ordered_embeddings = [None] * len(new_perm)
                for i, p in enumerate(new_perm):
                    ordered_embeddings[i] = all_embeddings[j][p]
                
                all_shuffled_embeddings.append(ordered_embeddings)         
            torch.save(all_shuffled_embeddings, os.path.join(save_directory_embeddings, f"table_{table_index}_embeddings.pt"))
            avg_cosine_similarities, mcvs, table_avg_cosine_similarity, table_avg_mcv = analyze_embeddings(all_shuffled_embeddings)
            results = {
                "avg_cosine_similarities": avg_cosine_similarities,
                "mcvs": mcvs,
                "table_avg_cosine_similarity": table_avg_cosine_similarity,
                "table_avg_mcv": table_avg_mcv
            }
            print(f"Table {table_index}:")
            print("Average Cosine Similarities:", results["avg_cosine_similarities"])
            print("MCVs:", results["mcvs"])
            # print("Table Average Cosine Similarity:", results["table_avg_cosine_similarity"])
            # print("Table Average MCV:", results["table_avg_mcv"])
            torch.save(results, os.path.join(save_directory_results, f"table_{table_index}_results.pt"))
            # table_index = table_index + 1
            # print("=" * 50)
            # print(len(all_shuffled_embeddings))
            # print(all_shuffled_embeddings[0].shape)
            # print("=" * 50)
        
