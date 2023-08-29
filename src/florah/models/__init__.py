
from . import base_modules, flows, transforms
from . import rnn_model, attention_model


ALL_MODELS = {
    "AttentionMAF": attention_model.att_generator,
    "RecurrentMAF": rnn_model.rnn_generator,
}

def get_model_arch(model_arch):
     """ Return architecture given arch name """
     if model_arch not in ALL_MODELS:
         raise KeyError(
                 f"Unknown arch name \"{model_arch}\"."\
                 f"Available archs are: {str(ALL_MODELS.keys())}")
     return ALL_MODELS[model_arch]
