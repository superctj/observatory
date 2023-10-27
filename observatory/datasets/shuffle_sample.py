import random
import itertools
import math

def fisher_yates_shuffle(seq):
    for i in reversed(range(1, len(seq))):
        j = random.randint(0, i)
        seq[i], seq[j] = seq[j], seq[i]
    return seq


def get_permutations(n, m):
    if n < 10:
        # Generate all permutations
        all_perms = list(itertools.permutations(range(n)))
        # Remove the original sequence
        all_perms.remove(tuple(range(n)))
        # Shuffle the permutations
        random.shuffle(all_perms)
        # If m > n! - 1 (because we removed one permutation), return all permutations
        if m > len(all_perms):
            return [list(range(n))] + all_perms
        # Otherwise, return the first m permutations
        return [list(range(n))] + all_perms[:m]
    else:
        original_seq = list(range(n))
        perms = [original_seq.copy()]
        for _ in range(m):  # we already have one permutation
            while True:
                new_perm = fisher_yates_shuffle(original_seq.copy())
                if new_perm not in perms:
                    perms.append(new_perm)
                    break
        return perms

def get_subsets(n, m, portion):
    portion_size = int(n * portion)
    max_possible_tables = math.comb(n, portion_size)

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


def column_shuffle_df(df, m):
    # Get the permutations
    perms = get_permutations(len(df.columns), m)

    # Create a new dataframe for each permutation
    dfs = []
    for perm in perms:
        dfs.append(df.iloc[:, list(perm)])

    return dfs, perms

def row_shuffle_df(df, m):
    # Get the permutations
    perms = get_permutations(len(df), m)

    # Create a new dataframe for each permutation
    dfs = []
    for perm in perms:
        dfs.append(df.iloc[list(perm)])

    return dfs


def sample_df(df, m, portion):
    subsets = get_subsets(len(df), m, portion)
    dfs = [df]
    for subset in subsets:
        dfs.append(df.iloc[subset])
    return dfs