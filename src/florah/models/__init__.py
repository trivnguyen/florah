
from . import base_modules, flows, transforms
from . import recurrent_maf, attention_maf

__all__ = [
    "base_modules",
    "flows",
    "transforms",
    "recurrent_maf",
    "attention_maf",
]

ALL_MODELS = {
    "AttentionMAF": attention_maf,
    "RecurrentMAF": recurrent_maf,
}

def get_model_arch(model_arch):
     """ Return architecture given arch name """
     if model_arch not in ALL_MODELS:
         raise KeyError(
                 f"Unknown arch name \"{model_arch}\"."\
                 f"Available archs are: {str(ALL_MODELS.keys())}")
     return ALL_MODELS[model_arch]
