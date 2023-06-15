import argparse
from scipy.special import comb
import random
import itertools
import torch
from table_bert import Table, Column
from table_bert import TableBertModel
import numpy as np
import pandas as pd


def convert_to_table(df, tokenizer):

    header = []
    data = []

    for col in df.columns:
        try:
            # Remove commas and attempt to convert to float
            val = float(str(df[col].iloc[0]).replace(',', ''))
            # If conversion is successful, it's a real column
            col_type = 'real'
            sample_value = df[col][0]
        except (ValueError, AttributeError):
            # If conversion fails, it's a text column
            col_type = 'text'
            sample_value = df[col][0]

        # Create a Column object
        header.append(Column(col, col_type, sample_value=sample_value))
        
        # Add the column data to 'data' list
    for row_index in len(df):
        data.append(list(df.iloc[row_index]))
        # print()
        # print(col_type)
        # print(sample_value)
    # Create the Table
    table = Table(id='', header=header, data=data)

    # Tokenize
    table.tokenize(tokenizer)

    return table


def get_subsets(n, m, portion):

    portion_size = int(n * portion)

    max_possible_tables = comb(n, portion_size)

    if max_possible_tables <= 10 * m:

        # If the number of combinations is small, generate all combinations and randomly select from them

        all_subsets = list(itertools.combinations(range(n), portion_size))

        random.shuffle(all_subsets)

        return [list(subset) for subset in all_subsets[:m]]

    else:

        # If the number of combinations is large, use random sampling to generate distinct subsets
        subsets = set()

        while len(subsets) < min(m, max_possible_tables):

            new_subset = tuple(sorted(random.sample(range(n), portion_size)))

            subsets.add(new_subset)

        return [list(subset) for subset in subsets]


def shuffle_df(df, m, portion):
    subsets = get_subsets(len(df), m, portion)

    dfs = [df]

    for subset in subsets:

        dfs.append(df.iloc[subset].copy())
    return dfs


model = TableBertModel.from_pretrained(
    '/home/zjsun/TaBert/TaBERT/tabert_base_k3/model.bin',
)
# print(torch.version.cuda)
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
model.eval()
# print(np.random.randint(20, 70, size=200))
# data = {
#     "Name": [ str(i) for i in range(1, 201)],
#     "Age": np.random.randint(20, 70, size=200),
#     "City": [ str(np.random.randint(1, 5)) for _ in range(200)]
# }
# original_header=[
#             Column('level', 'text', sample_value='A'),
#             Column('team', 'text', sample_value='Hartford Bees'),
#             Column('league', 'text', sample_value='Eastern League '),
#             Column('manager', 'text', sample_value='Jack Onslow'),
#         ]
# original_data=[
#             ['A', 'Hartford Bees', 'Eastern League', 'Jack Onslow'],
#             ['B', 'Evansville Bees', 'Illinois-Indiana-Iowa League', 'Bob Coleman'],
#             ['B', 'York Bees', 'Interstate League', 'Rudy Hulswitt'],
#             ['D', 'Owensboro Oilers', 'KITTY League', 'Hughie Wise and Harold Sueme'],
#             ['D', 'Bradford Bees', 'PONY League', 'Eddie Onslow and Vic George']
#         ]


def get_subsets(n, m, portion):

    portion_size = int(n * portion)

    max_possible_tables = comb(n, portion_size)

    if max_possible_tables <= 10 * m:

        # If the number of combinations is small, generate all combinations and randomly select from them

        all_subsets = list(itertools.combinations(range(n), portion_size))

        random.shuffle(all_subsets)

        return [list(subset) for subset in all_subsets[:m]]

    else:

        # If the number of combinations is large, use random sampling to generate distinct subsets
        subsets = set()

        while len(subsets) < min(m, max_possible_tables):

            new_subset = tuple(sorted(random.sample(range(n), portion_size)))

            subsets.add(new_subset)

        return [list(subset) for subset in subsets]


def shuffle_lists(lists, perms):
    # Add the original list to the shuffled lists
    shuffled_lists = [lists.copy()]
    # Apply permutations to lists
    for perm in perms:
        shuffled_list = [lists[i] for i in perm]
        shuffled_lists.append(shuffled_list)
    return shuffled_lists


parser = argparse.ArgumentParser(

    description='Process tables and save embeddings.')
parser.add_argument('-r', '--read_directory', type=str,

                    required=True, help='Directory to read tables from')


parser.add_argument('-t', '--table_num', type=int,

                    default=0, help='num of start table')
# parser.add_argument('-k', '--kaishi', type=int,

#                     default=0, help='num of start table')
# parser.add_argument('-j', '--jieshu', type=int,

#                     default=0, help='num of start table')

args = parser.parse_args()

table = pd.read_csv(

    f"{args.read_directory}/table_{args.table_num}.csv", keep_default_na=False)
k = 20
table = table.iloc[:, 10:15]

portion = 0.25
num_shuffles = 1000
tables = shuffle_df(table, num_shuffles, portion)
for j in range(len(tables)):
    processed_table = tables[j]
    try:

        processed_table = processed_table.reset_index(drop=True)

        processed_table = processed_table.astype(str)
        processed_table = convert_to_table(processed_table, model.tokenizer)
        context = ''
        with torch.no_grad():
            context_encoding, column_encoding, info_dict = model.encode(
                contexts=[model.tokenizer.tokenize(context)],
                tables=[processed_table]
            )
        embeddings = column_encoding[0]

        # Free up some memory by deleting column_encoding and info_dict variables
        del column_encoding
        del info_dict
        del context_encoding
        del embeddings
        # Empty the cache
        torch.cuda.empty_cache()
    except Exception as e:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        with open('demo_message.txt', 'a') as f:
            # Write the exception into the file
            f.write(str(e) + '\n')

            # Write the table columns into the file
            f.write(', '.join(map(str, tables[j].columns)) + '\n')

            # Write the table into the file
            f.write(str(tables[j]) + '\n\n')
        # print("Error message:", e)

        # print(tables[j].columns)
        # print(tables[j])
# perms = get_subsets(len(original_header), 1000, portion)
# headers = shuffle_lists(original_header, perms)
# list_data = shuffle_lists(original_data, perms)
# for header, data in zip(headers, list_data):
#     try:
#         table = Table(
#                 id='',
#                 header=header,
#                 data=data
#             ).tokenize(model.tokenizer)


#         context = ''

        # model takes batched, tokenized inputs
        # with torch.no_grad():

        #     context_encoding, column_encoding, info_dict = model.encode(
        #         contexts=[model.tokenizer.tokenize(context)],
        #         tables=[table]
        #     )
        #     print(column_encoding)

    # print("context_encoding :", context_encoding)
    # print("column_encoding shape:", column_encoding.shape)

    # print("column_encoding[0] :", column_encoding[0])

# df = pd.DataFrame(data)


# print("info_dict :", info_dict)
# table = convert_to_table(df, model.tokenizer)
# table = Table(
#     id='',
#     header=[
#         Column('Nation', 'text', sample_value='United States'),
#         Column('Gross Domestic Product', 'real', sample_value='21,439,453')
#     ],
#     data=[
#         ['United States' + str(i) for i in range(1, 501)],
#         np.random.randint(20, 70, size=500),

#     ]
# ).tokenize(model.tokenizer)

# To visualize table in an IPython notebook:
# display(table.to_data_frame(), detokenize=True)
