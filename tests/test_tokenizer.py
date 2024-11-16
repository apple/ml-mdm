# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import logging

from pathlib import Path    
from ml_mdm.language_models.tokenizer import Tokenizer # Tokenizer class from tokenizer.py

def test_tokenizer_bert():
    f = Path(__file__).parent/"data/bert.vocab"     # To solve from relative to absolute import
    assert Tokenizer(f, mode="bert")

def test_tokenizer_t5():
    f = Path(__file__).parent/"data/t5.vocab"   
    assert Tokenizer(f, mode="tf")
    
def test_tokenizer():
    f = Path(__file__).parent/"data/imagenet.vocab"   
    assert Tokenizer(f)

test_tokenizer_bert()
test_tokenizer_t5()
test_tokenizer()
