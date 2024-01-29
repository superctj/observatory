import gzip
import codecs
# import ujson as json
import numpy as np
import re

RE_REGULARIZE_NUMBERS = re.compile('[0-9]')
RE_REGULARIZE_SPECIAL_SIGNS = re.compile('[\.!\?*\\\\/+\-:\;\'"#~<>`%$&@€\[\]{}]')
VALID_CHARS_SET_WITH_SPECIAL_SINGS = set(
    '!?*\\/+-:;\'"#~<>`%$&@€[]{}0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz., ')
RE_VALID_STRING = re.compile('[a-zA-Z0-9,\s\.!\?*\\\\/+\-:\;\'"#~<>`%$&@€\[\]{}]*')

def load_termlist(termlist_path):
    f = open(termlist_path, 'r')
    terms = dict()
    for i, line in enumerate(f):
        terms[i] = line[:-1]
    return terms

def split_attribute(attribute):
    """Splits a list of attributes in head and remainder.
    """
    return attribute[0], attribute[1:]


def replace_whitespaces(term):
    # TODO check if this function has a use case
    return term.replace('\\', '\\\\').replace('_', '\\_').replace(' ', '_')


def reconstruct_whitespaces(term):
    # TODO check if this function has a use case
    result = ''
    escape = False
    for i in range(len(term)):
        if term[i] == '\\':
            if escape:
                result += '\\'
                escape = False
                continue
            else:
                escape = True
                continue
        if term[i] == '_':
            if escape:
                result += '_'
                escape = False
                continue
            else:
                result += ' '
                continue
        result += term[i]
    return result


# def load_index_file(filename):
#     f = gzip.open(filename, 'r')
#     reader = codecs.getreader('utf-8')
#     stream = reader(f)
#     data = json.load(stream)
#     return data

def load_embedding_file(embedding_path):
    """Loads a word or node embedding file and parse its content to a dict object.

    Returns:
        vectors: dictionary mapping words / nodes to vectors
        d: dimensionality of vectors
    """
    f = open(embedding_path, 'r')
    size, d = f.readline().split()
    vectors = dict()
    for line in f:
        data = line.split()
        try:
            vectors[data[0]] = np.array(data[1:], dtype='float32')
        except:
            print('Can not parse line:', line)

    return vectors, int(d)


def get_column_term(url, header):
    # TODO check if this function has a use case
    return url.replace('~', '\\~').replace('\\', '\\\\') + '~' + header.replace('~', '\\~').replace('\\', '\\\\')


def parse_column_term(term):
    # TODO check if this function has a use case
    is_escaped = False
    url = ''
    header = ''
    state = 0
    for c in term:
        if c == '~':
            if is_escaped:
                if state == 0:
                    url += c
                else:
                    header += c
                is_escaped = False
            else:
                state = 1
                continue
        if c == '\\':
            if is_escaped:
                is_escaped = False
                if state == 0:
                    url += c
                else:
                    header += c
            else:
                is_escaped = True
                continue
        is_escaped = False
        if state == 0:
            url += c
        else:
            header += c
    return url, header