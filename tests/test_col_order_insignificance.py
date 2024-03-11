import math

from collections import Counter

from observatory.properties.Column_Order_Insignificance.evaluate_col_shuffle import (  # noqa: E501
    fisher_yates_shuffle,
    get_permutations,
)


def test_fisher_yates_shuffle():
    sequence = list(range(10))
    shuffled_seq = fisher_yates_shuffle(sequence)
    assert Counter(sequence) == Counter(shuffled_seq)


def test_get_permutations():
    n = 3
    m = 1
    permutations = get_permutations(n, m)
    assert len(permutations) == min(m + 1, math.factorial(n))

    n = 3
    m = 5
    permutations = get_permutations(n, m)
    assert len(permutations) == min(m + 1, math.factorial(n))

    n = 3
    m = 6
    permutations = get_permutations(n, m)
    assert len(permutations) == min(m + 1, math.factorial(n))

    n = 3
    m = 7
    permutations = get_permutations(n, m)
    assert len(permutations) == min(m + 1, math.factorial(n))

    n = 10
    m = 10
    permutations = get_permutations(n, m)
    assert len(permutations) == min(m + 1, math.factorial(n))

    uniq_permut = set()
    for permut in permutations:
        permut = tuple(permut)

        if permut not in uniq_permut:
            uniq_permut.add(permut)
        else:
            raise ValueError(f"Permutation {permut} is not unique.")


def test_shuffle_df_columns():
    pass
