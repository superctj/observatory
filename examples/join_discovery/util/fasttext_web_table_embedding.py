import sys
import os
import re
import json
import gzip
import math
import random
import importlib
import numpy as np
# import networkx as nx
from collections import defaultdict
# from whatthelang import WhatTheLang

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# if importlib.util.find_spec('fastText') != None:
#     import fastText
# else:
#     import fasttext as fastText
import fasttext as fastText

import util.wte_util as wte_util


class FastTextWebTableModel:
    def __init__(self, config=None, model=None, create_walks=True):
        OFFSET_1 = -60
        OFFSET_2 = -61

        self.VALID_CHARS = np.array([x for x in range(128)],
                                    dtype='byte').tobytes().decode()

        self.CODES = np.array([OFFSET_1 if (x % 2 == 0) and (x < 128) else
                               (OFFSET_2 if x % 2 == 0 else 128 + ((x - 1) / 2) % 64) for x in range(256)],
                              dtype='byte')

        self.code2char, self.char2code = self._construct_codec()

        if model == 'dummy':
            model = None
            print('Created dummy model')
        elif model != None:
            self.model = model
        elif config != None:
            self.train_fasttext_model(config, create_walks=create_walks)
        else:
            raise Exception('FastTextWebTableModel needs to be initialized'
                            'either with by a pre-trained FastText Model or'
                            'with a training configuration file!')

    @staticmethod
    def load_model(filename):
        model = FastTextWebTableModel(model=fastText.load_model(filename))
        return model

    def train_fasttext_model(self, config, create_walks=True):
        if create_walks:
            print('Create random walks ...')
            self.create_walks(config)
        print('Start Training ...')
        self.model = fastText.train_unsupervised(
            config['walks_filename'], dim=config['dim'], minCount=config['min_count'], ws=10, neg=5, epoch=1, maxn=7, lr=float(config['lr']), lrUpdateRate=int(config['lrUpdateRate']), thread=24)
        return

    def save_model(self, path):
        self.model.save_model(path)

    def get_dimension(self):
        return self.model.get_dimension()

    def get_header_vector(self, term):
        if len(term) == 0:
            term = '_'
        return self.model.get_word_vector(self.encode_header(term))

    def get_data_vector(self, term):
        if type(term) != str or len(term) == 0: # there is a hddien nan value in the table
            term = '_'
        # try:
        #     if len(term) == 0:
        #         term = '_'
        # except TypeError:
        #     print("\n ======")
        #     print(term)
        #     print("======\n")
        #     raise TypeError
        return self.model.get_word_vector(self.encode_data(term))

    def get_plain_vector(self, term):
        return self.model.get_word_vector(term)

    def create_walks(self, config):
        if config['walk_type'] == 'base':
            self._create_row_walks(config)
        elif config['walk_type'] == 'row':
            self._create_row_walks(config)
        elif config['walk_type'] == 'tax':
            self._create_tax_walks(config)
            self._encode_tax_walks(config)
        elif config['walk_type'] == 'tax-from-ids':
            self._encode_tax_walks(config)
        elif config['walk_type'] == 'combo':
            self._create_row_walks(config)
            self._create_tax_walks(config)
            self._encode_tax_walks(config, append=True)
        else:
            raise Exception('Unkown walks_type: ' + config['walk_type'])
        return

    def _create_tax_walks(self, config, append=False):
        MAXIMAL_TRIALS_FACTOR = 20
        MAXIMAL_TRIALS = config['walk_length'] * MAXIMAL_TRIALS_FACTOR
        id_walks_filename = config['walks_filename'] + '.ids'
        node_index = None
        choice_function = None
        node_index = self._load_weighted_graph(config)
        choice_function = self._weighted_choice
        nodes = list(node_index.keys())
        id_walks_file = open(id_walks_filename, 'a') if append else open(
            id_walks_filename, 'w')
        for i in range(config['number_walks']):
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                blacklist = set()
                c = 0
                while (len(walk) < config['walk_length']) and (c < MAXIMAL_TRIALS):
                    if len(node_index[walk[-1]]) > 0:
                        next = choice_function(node_index[walk[-1]])
                        c += 1
                        if next not in blacklist:
                            walk.append(next)
                            blacklist.add(next)
                    else:
                        break
                id_walks_file.write(
                    ' '.join([str(n) for n in walk]) + os.linesep)
        id_walks_file.close()
        return id_walks_filename

    def _encode_tax_walks(self, config, append=False):
        DATA_PREFIX = 'd#'
        HEADER_PREFIX = 'h#'
        term_list = wte_util.load_termlist(
            config['termlist_path'])  # dictionary id -> term
        id_walks_filename = config['walks_filename'] + '.ids'
        f_ids = open(id_walks_filename, 'r')
        f_terms = open(config['walks_filename'], 'a') if append else open(
            config['walks_filename'], 'w')

        for line in f_ids:
            terms_of_walk = []
            for id in line.split():
                graph_term = term_list[int(id)]
                graph_prefix = graph_term[:2]
                if graph_prefix == HEADER_PREFIX:
                    terms_of_walk.append(self.encode_header(
                        graph_term[2:], has_wildcards=True))
                elif graph_prefix == DATA_PREFIX:
                    terms_of_walk.append(self.encode_data(
                        graph_term[2:], has_wildcards=True))
                else:
                    print('Unknown graph term prefix: ',
                          graph_prefix, file=sys.stderr)
            new_line = ' '.join(terms_of_walk)
            f_terms.write(new_line + os.linesep)
        f_ids.close()

        f_terms.close()
        return

    def _create_row_walks(self, config, append=False, size=float('inf')):
        BATCH_SIZE = 1000
        # wtl = WhatTheLang()
        f = gzip.open(config['dump_path'], 'rt', encoding='utf-8')
        meta_data = f.readline()
        line = f.readline()
        walks = []
        count = 0
        f_out = open(config['walks_filename'], 'a') if append else open(
            config['walks_filename'], 'w')
        while line:
            count += 1
            if count % 1000 == 0:
                print('Processed', count, 'tables')
            if count > size:
                break
            try:
                data = json.loads(line)
            except:
                print('Can not parse:', count, line)
                line = f.readline()
            walks += self._create_row_walks_from_table(config, data)
            line = f.readline()
            if len(walks) > BATCH_SIZE:
                for walk_line in walks:
                    f_out.write(walk_line + os.linesep)
                walks = []
        for walk_line in walks:
            f_out.write(walk_line + os.linesep)
        f.close()
        f_out.close()
        return

    def _create_row_walks_from_table(self, config, table, wtl=None):
        if table['headerPosition'] != 'FIRST_COLUMN':
            table['relation'] = list(zip(*table['relation']))
        walks = []
        if len([x for x in table['relation'][0] if (x != None) and (len(x) > 0)]) < config['min_columns']:
            return []
        # if config['lang_filter'] != 'none':
        #     text = ' '.join([' '.join([r for r in row if r != None]) for i, row in enumerate(
        #         table['relation']) if i < float(config['max_rows'])])
            # lang = wtl.predict_lang(text)
            # if lang != config['lang_filter']:
            #     return []
        if config['walk_type'].lower() == 'base':
            for row in table['relation']:
                walks.append(
                    ' '.join([x for x in row if (x != None) and (len(x) > 0)]))
        elif (config['walk_type'].lower() == 'row') or (config['walk_type'].lower() == 'combo'):
            for i, row in enumerate(table['relation']):
                if i >= float(config['max_rows']):
                    break
                row = [x for x in row if (x != None) and (len(x) > 0)]
                walk = []
                for elem in row:
                    elem = elem.replace('\n', '')
                    if i == 0:
                        encoded_elem = self.encode_header(elem)
                    else:
                        encoded_elem = self.encode_data(elem)
                    if len(encoded_elem) > 0:
                        walk.append(encoded_elem)
                if len(walk) >= config['min_columns']:
                    walks.append(' '.join(walk))
        else:
            print('Unknown walk type:', config['walk_type'])
        return walks

    def encode_header(self, text_value, has_wildcards=False):
        value = self._clear_string_with_wildcards(
            text_value) if has_wildcards else self._clear_string(text_value)
        result = ''
        for c in value:
            result += self.char2code[c]
        return result

    def encode_data(self, text_value, has_wildcards=False):
        if has_wildcards:
            return self._clear_string_with_wildcards(text_value)
        else:
            return self._clear_string(text_value)

    def decode_header(self, text_value):
        return ''.join([self.code2char[c] for c in text_value])

    def decode_data(self, text_value):
        return text_value

    def _clear_string(self, text_value):
        result = ''
        for c in text_value:
            if c in wte_util.VALID_CHARS_SET_WITH_SPECIAL_SINGS:
                result += c
        return wte_util.RE_REGULARIZE_NUMBERS.sub('@', wte_util.RE_REGULARIZE_SPECIAL_SIGNS.sub('*', result.replace(' ', '_'))).lower()

    def _clear_string_with_wildcards(self, text_value):
        result = ''
        for c in text_value:
            if c in wte_util.VALID_CHARS_SET_WITH_SPECIAL_SINGS:
                result += c
        return result.replace(' ', '_').lower()

    @staticmethod
    def _weighted_choice(neighbors):
        (neighbors, s) = neighbors
        r = random.random() * s
        search_space = (0, len(neighbors))
        while search_space[1] - search_space[0] > 1:
            m = int((search_space[0] + search_space[1]) / 2)
            if neighbors[m][2] < r:
                search_space = (m, search_space[1])
            else:
                search_space = (search_space[0], m)
        return neighbors[search_space[0]][0]

    def _load_weighted_graph(self, config):
        f = open(config['edgelist_path'], 'r')
        index = defaultdict(dict)
        for line in f:
            u, v, w = line.split(' ')
            u, v, w = int(u), int(v), float(w)
            if w < config['weight_limit']:
                continue
            index[u][v] = w
            index[v][u] = w
        for key in index:
            index[key] = [(x, y) for (x, y) in index[key].items()]
        for key in index:
            s = sum([x[1] for x in index[key]])
            w_sum = 0
            new_elem = []
            for (x, y) in index[key]:
                w_sum += y
                new_elem.append((x, y, w_sum))
            index[key] = (new_elem, sum([x[1] for x in index[key]]))
        return index

    def _construct_codec(self):
        def get_code(x): return self.CODES[x * 2:x * 2 + 2].tobytes().decode()
        id_lookup = dict([(i, v) for i, v in enumerate(self.VALID_CHARS)])
        code2char = dict([(get_code(i), id_lookup[i])
                          for i in range(len(self.VALID_CHARS))])
        char2code = dict([(id_lookup[i], get_code(i))
                          for i in range(len(self.VALID_CHARS))])
        return code2char, char2code


def create_arg_parser():
    parser = ArgumentParser("fasttext_web_table_embeddings",
                            description="Create web table embeddings from a header data web table graph using a fasttext model",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument(
        '-c', '--config', help="configuration file with the training parameters", nargs=1, required=True)
    parser.add_argument(
        '-o', '--output', help="output path to store the embedding model in the fastText text format", nargs=1, required=True)
    parser.add_argument(
        '-w', '--walks', help="use already existing walks file", nargs='?', const=True, default=False)

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    # Parse config file
    f_config = open(args.config[0], 'r')
    config = json.load(f_config)
    model = FastTextWebTableModel(config=config, create_walks=args.walks)

    print('Save model to disk ...')
    model.save_model(args.output[0])

    return


if __name__ == "__main__":
    main()