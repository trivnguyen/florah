
from . import bijectors
from . import flows
from . import transforms
from . import modules
from . import recurrent_maf
from . import attention_maf
from . import recurrent_maf_decay

ALL_MODELS = {
    "AttentionMAF": attention_maf,
    "RecurrentMAF": recurrent_maf,
    "RecurrentMAFDecay": recurrent_maf_decay,
}

def get_model_arch(model_arch):
     """ Return architecture given arch name """
     if model_arch not in ALL_MODELS:
         raise KeyError(
                 f"Unknown arch name \"{model_arch}\"."\
                 f"Available archs are: {str(ALL_MODELS.keys())}")
     return ALL_MODELS[model_arch]
