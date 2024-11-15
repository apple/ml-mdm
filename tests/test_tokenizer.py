# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import logging

# Tokenizer class from tokenizer.py
from pathlib import Path    
from ml_mdm.language_models.tokenizer import Tokenizer # Tokenizer class from tokenizer.py

# Vocab files are in /ml-mdm/data/

# Tokenizer.py: each function is a different mode of tokenizing a token_file (?)
#   (is this the same as a vocab file or is it a dictionary of tokens?) and they
#   return a prefix tree structure.


def test_tokenizer_bert():
    # f = "../data/bert.vocab" 
    f = Path(__file__).parent/"data/bert.vocab"     # To solve from relative to absolute import
    assert Tokenizer(f, mode="bert")
    #Q: should we assert the contents of tokenizer?

def test_tokenizer_t5():
    f = "../data/t5.vocab" 
    assert Tokenizer(f, mode="tf")
    #Q: should we assert the contents of tokenizer?

# any vocab file that's not either bert or t5
def test_tokenizer():
    f = "../data/imagenet.vocab" 
    assert Tokenizer(f)
    #Q: should we assert the contents of tokenizer?

test_tokenizer_bert()
test_tokenizer_t5()
test_tokenizer()
