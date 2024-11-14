# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
from ml_mdm.language_models import tokenizer

# Vocab files are in /ml-mdm/data/
# Difference between .vocab file to a token file? 

# Tokenizer in factory.py create_tokenizer(vocab_file): you pass a vocab file, 
#   function returns something (dictionary? a tokenizer?)
# Tokenizer.py: each function is a different mode of tokenizing a token_file (?)
#   (is this the same as a vocab file or is it a dictionary of tokens?) and they
#   return a prefix tree structure.

# CASE 1: We compare factory.py's create_tokenizer() to tokenizer.py's output.
#  Ex.: assert create_tokenizer() == read_dictionary_bert()

# CASE 2: We run the output of factory, and then pass its output as the token 
#   file to tokenizer.py's functions. Then, we need something else to assert 
#   against.
#  Ex.: read_dictionary_bert(create_tokenizer(.vocab)) == ?

def test_tokenizer_bert():
    pass

def test_tokenizer_t5():
    pass

# Is this a general vocab file that's not either bert or t5?
def test_tokenizer():
    pass


