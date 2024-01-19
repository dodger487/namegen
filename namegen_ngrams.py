# %%
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# %%
df = pd.read_csv("data.csv")

# %%
df.head()

# %%
df.shape

# %%
small = df[(df.year > 2010) & (df.gender == "M")]
small["clean_name"] = '^' + small.name.str.lower() + '$'


# %%
small.shape

# %%
small.head()

# %%
def get_name_transition_probs(name):
    return Counter((name[i], name[i+1]) for i in range(len(name) - 1))

# %%
def get_n_grams(text, n):
    """
    Returns a list of n-grams from a text.
    """
    n_grams = []
    for i in range(len(text) - n + 1):
        n_grams.append(text[i:i + n])
    return n_grams

# %%
get_n_grams("^baba$", 2)

# %%
class TransitionCount(defaultdict):
    def __init__(self, *args, **kwargs):
        super(TransitionCount, self).__init__(Counter)
        self.update(*args, **kwargs)

    def __add__(self, other):
        all_keys = self.keys() | other.keys()
        return TransitionCount({k: self[k] + other[k] for k in all_keys})
    
    def __mul__(self, scalar):
        new = TransitionCount()
        for k in self.keys():
            for k2 in self[k].keys():
                new[k][k2] = self[k][k2] * scalar
        return new

# %%
def get_transition_counts(text):
    out_dict = TransitionCount()
    for i in range(len(text) - 1):
        out_dict[text[i]][text[i + 1]] += 1
    return out_dict

# %%
get_transition_counts("^babc$")

# %%
get_transition_counts("mary").keys() | get_transition_counts("john").keys()

# %%
get_transition_counts("mary") + get_transition_counts("mom")


# %%
get_transition_counts("mary") * 10

# %%
small.head()

# %%
cnts = [get_transition_counts(c) * freq for c, freq in zip(small.clean_name, small["count"])]

# %%
cnts[1]

# %%
all_transition_counts = sum(cnts, start=TransitionCount())

# %%
all_transition_counts["c"]

# %%
def normalize_transition_count(transition_count):
    """
    This function normalizes the transition count dictionary by dividing each
    count by the total count for the first word in the transition.
    """
    for letter, next_letters in transition_count.items():
        total_count = sum(next_letters.values())
        for l in next_letters:
            next_letters[l] /= total_count

# %%
normalize_transition_count(all_transition_counts)

# %%
all_transition_counts["c"]

# %%
all_transition_counts["^"]

# %%
def generate_name(transition_probs, n=1):
    name = "^"
    while name[-1] != "$":
        next_char = np.random.choice(list(transition_probs[name[-n:]].keys()),
                                     p=list(transition_probs[name[-n:]].values()))
        name += next_char
    return name[1:-1]


# %%
generate_name(all_transition_counts)

# %%
def get_transition_counts2(text):
    out_dict = TransitionCount()
    for i in range(len(text) - 2):
        out_dict[text[i:i+2]][text[i + 2]] += 1
    return out_dict

# %%
get_transition_counts2("^chris$")

# %%
all_transition_counts2 = sum((get_transition_counts2(c) * freq 
                             for c, freq in zip(small.clean_name, small["count"]))
                            , start=TransitionCount())

# %%
all_transition_counts2["ch"]

# %%
normalize_transition_count(all_transition_counts2)

# %%
generate_name(all_transition_counts2, 2)

# %%
import random

# %%
start_states = [k for k in all_transition_counts2.keys() if k[0] == '^']

# %%
n=2
# name = "^j"
name = random.choice(start_states)
while name[-1] != "$":
    next_char = np.random.choice(list(all_transition_counts2[name[-n:]].keys()),
                                    p=list(all_transition_counts2[name[-n:]].values()))
    name += next_char
# return name[1:-1]
print(name[1].upper() + name[2:-1])

# %%
# start_probs = TransitionCount((k, all_transition_counts2[k]) for k in all_transition_counts2.keys() if k[0] == '^')


# %%


# %%
df.year.max()

# %%



