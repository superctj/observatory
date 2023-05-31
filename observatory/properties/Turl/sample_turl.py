import torch
import torch.nn as nn
import os

from TURL.model.configuration import TableConfig
from TURL.model.model import HybridTableMaskedLM
from analyze_embeddings import analyze_embeddings
# from observatory.models.transformers import load_transformers_tokenizer
import threading
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


if __name__ == "__main__":
    from sample_turl_wiki_tables import TurlWikiTableDataset
    from TURL.data_loader.CT_Wiki_data_loaders import CTLoader
    from TURL.model.transformers import BertTokenizer
    from TURL.utils.util import load_entity_vocab
    import argparse

    data_dir = ""
    min_ent_count = 2

    entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=min_ent_count)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = "table-base-config_v2.json"
    ckpt_path = "pytorch_model.bin"
    device = torch.device("cuda:0")
    parser = argparse.ArgumentParser(description='Process tables and save embeddings.')
    parser.add_argument('-s', '--save_directory', type=str, required=True, help='Directory to save embeddings to')
    parser.add_argument('-p', '--sample_portion', type=float, default=0.25, help='Portion of sample to use')
    parser.add_argument('-b', '--batch_size', type=int, default=13, help='Batch Size')
    parser.add_argument('-l', '--start_line', type=int, default=0, help='The index of start table')
    args = parser.parse_args()
    
    model = TURL(config, ckpt_path)
    model.to(device)
     # save_directory_results  = os.path.join('/nfs/turbo/coe-jag/zjsun', 'sample_portion', str(args.sample_portion), args.save_directory, model_name ,'results')
    # save_directory_embeddings  = os.path.join('/nfs/turbo/coe-jag/zjsun','sample_portion', str(args.sample_portion), args.save_directory, model_name ,'embeddings')
    save_directory_results  = os.path.join('/nfs/turbo/coe-jag/zjsun', 'sample_portion', str(args.sample_portion), args.save_directory, 'Turl' ,'results')
    save_directory_embeddings  = os.path.join('/nfs/turbo/coe-jag/zjsun', 'sample_portion', str(args.sample_portion), args.save_directory, 'Turl' ,'embeddings')
    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
    with open(os.path.join(data_dir, "test_tables.jsonl"), "r") as f:
        lines = f.readlines()
        for table_index in range(args.start_line, len(lines)):
            line = lines[table_index]
            
            
            # timer_flag = [False]
            # timer_thread = threading.Thread(target=set_timer, args=(timer_flag,))
            # timer_thread.start()

            # try:
            #     test_dataset = TurlWikiTableDataset(line, entity_vocab, tokenizer, split="test", force_new=False, portion = args.sample_portion)
            # except Exception as e:
            #     print(f"table{table_index} failed.")
            #     print(f"Error: {e}")
            #     continue  # if an error occurs, continue to the next line

            # while timer_thread.is_alive():
            #     time.sleep(1)  # check every second

            # if timer_flag[0]:
            #     print("Operation took too long, moving to next iteration.")
            #     continue
            
            
            test_dataset = TurlWikiTableDataset(line, entity_vocab, tokenizer, split="test", force_new=False, portion = args.sample_portion)    
                
                
                
            if len(test_dataset) < 2:
                continue
            test_dataloader = CTLoader(test_dataset, batch_size=args.batch_size, is_train=False)
            all_shuffled_embeddings = []
            


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
                    
                    if all_shuffled_embeddings == []:
                        all_shuffled_embeddings = col_embeddings
                        # print(all_shuffled_embeddings.size())
                    else:
                        # print(all_shuffled_embeddings.size())
                        all_shuffled_embeddings = torch.cat((all_shuffled_embeddings, col_embeddings), dim=0)
                        
                    
                    
                    
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
            print("Table Average Cosine Similarity:", results["table_avg_cosine_similarity"])
            print("Table Average MCV:", results["table_avg_mcv"])
            torch.save(results, os.path.join(save_directory_results, f"table_{table_index}_results.pt"))
            # table_index = table_index + 1
            # print("=" * 50)
            # print(len(all_shuffled_embeddings))
            # print(all_shuffled_embeddings[0].shape)
            # print("=" * 50)
        
