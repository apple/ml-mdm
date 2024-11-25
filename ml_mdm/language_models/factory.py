# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import logging

from transformers import T5ForConditionalGeneration

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_mdm.language_models.tokenizer import Tokenizer     


class T5Encoder(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def embed_dim(self):
        return self.model_dim

    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_dict=None,
        return_penultimate=None,
        *args,
        **kwargs
    ):
        output = self.encoder.forward(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            *args,
            **kwargs
        )
        return output.last_hidden_state

    def load(self):
        pass


class LanguageModel(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.embed_dim = model.embed_dim
        self.device = "cpu"
        self.args = args

        # use pre-computed text embeddings. delete the language model!
        if args.use_precomputed_text_embeddings:
            del self.model
            self.model = None
            logging.info("<----------- delete the language model. ------------>")

    def to(self, device):
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self

    def forward(self, sample, tokenizer):
        lm_outputs, lm_mask, args = None, None, self.args
        if not isinstance(sample["tokens"], torch.Tensor):
            sample_tokens = (
                torch.from_numpy(sample["tokens"]).to(self.device).type(torch.long)
            )
        else:
            sample_tokens = sample["tokens"]

        if args.categorical_conditioning:
            lm_outputs = (
                F.one_hot(
                    sample_tokens[
                        :, 1
                    ],  # FIXME: we have bos token now (check reader_config?)
                    num_classes=tokenizer.vocab_size,  # vocab_size is actually num_classes + 3 but we don't worry about it for now
                )
                .type(torch.float32)
                .unsqueeze(1)
            )
        else:
            PAD_TOKEN = tokenizer.token_id(args.reader_config.padding_token)
            lm_mask = (sample_tokens != PAD_TOKEN).float()
            if args.use_precomputed_text_embeddings:
                lm_outputs = sample["text_embedding"].float()
            else:
                # execute language model
                if args.fp16:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        lm_outputs = self.model(
                            sample_tokens, lm_mask, return_penultimate=True
                        ).float()
                else:
                    lm_outputs = self.model(
                        sample_tokens, lm_mask, return_penultimate=True
                    ).float()

            lm_outputs = lm_outputs * lm_mask.unsqueeze(-1)
        return lm_outputs, lm_mask


def init_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_init(m: nn.Module):
        # - check if the current module has init_parameters & if it's callabed called it on m
        init_parameters = getattr(m, "init_parameters", None)
        if callable(init_parameters):
            m.init_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_init)


def create_tokenizer(vocab_file):
    return Tokenizer(vocab_file, mode="t5")


def create_lm(args, device: torch.device = "cuda"):
    tokenizer, model, language_model_dim = None, None, -1
    if args.categorical_conditioning:
        raise Exception("Not fixed yet, tokenizers were removed.")
        language_model_dim = (
            tokenizer.vocab_size
        )  # TODO (jack_carlson) this line is never reached
    else:
        tokenizer = create_tokenizer(args.vocab_file)
        model = T5Encoder.from_pretrained(args.text_model)
        model = LanguageModel(args, model).to(device)
    model.eval()
    return tokenizer, model
