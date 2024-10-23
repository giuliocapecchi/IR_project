#!pip install torch matplotlib nltk tqdm gdown

####################################################################

"""
# 1. Download the collection
"""

#import gdown

#url = 'https://drive.google.com/uc?id=1_wXJjiwdgc9Kpt7o7atP8oWe-U4Z56hn'
#gdown.download(url, 'collection.tsv', quiet=False)

####################################################################

"""
read 'collection.tsv' file and prepare it for data manipulation
the file is organized in the following way:
<pid>\t<text>\n
where <pid> is the passage id and <text> is the passage text
"""
import pandas as pd

df = pd.read_csv('collection.tsv', sep='\t', header=None)

####################################################################

# let's not truncate Pandas output too much
pd.set_option('display.max_colwidth', 50) # mettici 150
df.columns = ['doc_id', 'text']
print(df.head(2)) # returns the first N rows

####################################################################

import re
import string
import nltk

nltk.download("stopwords", quiet=True)

def preprocess(s):
    # lowercasing
    s = s.lower()
    # ampersand
    s = s.replace("&", " and ")
    # special chars
    s = s.translate(dict([(ord(x), ord(y)) for x, y in zip("‘’´“”–-", "'''\"\"--")]))
    # acronyms
    s = re.sub(r"\.(?!(\S[^. ])|\d)", "", s) # remove dots that are not part of an acronym
    # remove punctuation
    s = s.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # strip whitespaces
    s = s.strip()
    while "  " in s:
        s = s.replace("  ", " ")
    # tokeniser
    s = s.split()
    # stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    s = [t for t in s if t not in stopwords]
    # stemming
    stemmer = nltk.stem.PorterStemmer().stem
    s = [stemmer(t) for t in s]
    return s

####################################################################

import time

def profile(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        ms = (end - start) * 1000
        print(f"{f.__name__} ({ms:.3f} ms)")
        return result
    return f_timer

####################################################################

from collections import Counter
from tqdm.auto import tqdm

@profile
def build_index(dataset):
    lexicon = {}
    doc_index = []
    inv_d, inv_f = {}, {}
    termid = 0

    num_docs = 0
    total_dl = 0
    total_toks = 0
    for docid, doc in tqdm(enumerate(dataset.docs_iter()), desc='Indexing', total=dataset.docs_count()):
        tokens = preprocess(doc.text)
        #print(tokens)
        token_tf = Counter(tokens)
        for token, tf in token_tf.items():
            if token not in lexicon:
                lexicon[token] = [termid, 0, 0]
                inv_d[termid], inv_f[termid] =  [], []
                termid += 1
            token_id = lexicon[token][0] # prendo il termid
            inv_d[token_id].append(docid) # aggiungo il docid alla lista dei docid in cui compare il termine
            inv_f[token_id].append(tf) # aggiungo il tf alla lista dei tf in cui compare il termine
            lexicon[token][1] += 1 # incremento il df
            lexicon[token][2] += tf # tf è quanto compare il termine nel documento
        doclen = len(tokens)
        doc_index.append((str(doc.doc_id), doclen))
        total_dl += doclen
        num_docs += 1


    stats = {
        'num_docs': 1 + docid, # docid parte da 0
        'num_terms': len(lexicon),
        'num_tokens': total_dl,
    }
    return lexicon, {'docids': inv_d, 'freqs': inv_f}, doc_index, stats


####################################################################


"""
This class that takes the dataframe we created before with columns 'docno' and 'text', and creates a list of namedtuples
"""
from collections import namedtuple


class MSMarcoDataset:
    def __init__(self, df):
        self.docs = [Document(row.doc_id, row.text) for row in df.itertuples()]

    def docs_iter(self):
        return iter(self.docs)

    def docs_count(self):
        return len(self.docs)


Document = namedtuple('Document', ['doc_id', 'text']) # must define what a document is


####################################################################


# Test the MSMarcoDataset class by passing Document(1, "school"), Document(2, "example."), Document(3, "house.")

test_docs = [Document(1, "A school"), Document(2, "Another example."), Document(3, "This is a house.")]
test_dataset = MSMarcoDataset(pd.DataFrame(test_docs, columns=['doc_id', 'text']))

for doc in test_dataset.docs_iter():
    print(doc)

lex, inv, doc, stats = build_index(test_dataset)

####################################################################

print(lex)
print(inv)
print(doc)
print(stats)


####################################################################

# create a df with the first 10 rows
df = df.head(10) # TODO : REMOVE THIS

dataset = MSMarcoDataset(df)
lex, inv, doc, stats = build_index(dataset)


####################################################################

# create a df with the first 10 rows
df = df.head(10) # TODO : REMOVE THIS

dataset = MSMarcoDataset(df)
lex, inv, doc, stats = build_index(dataset)

####################################################################

print(lex)
print(inv)
print(doc)
print(stats)