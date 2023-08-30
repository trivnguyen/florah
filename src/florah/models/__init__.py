
from . import base_modules, flows, transforms
from .attention_model import att_generator
from .rnn_model import rnn_generator


ALL_MODELS = {
    "AttentionMAF": att_generator,
    "RecurrentMAF": rnn_generator,
}

def get_model_arch(model_arch):
     """ Return architecture given arch name """
     if model_arch not in ALL_MODELS:
         raise KeyError(
                 f"Unknown arch name \"{model_arch}\"."\
                 f"Available archs are: {str(ALL_MODELS.keys())}")
     return ALL_MODELS[model_arch]
