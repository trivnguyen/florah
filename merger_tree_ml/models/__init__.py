
from . import bijectors
from . import flows
from . import transforms
from . import modules
from . import recurrent_maf
from . import attention_maf

# arch_dict = {
#     "MAF": maf,
#     "RNN-MAF": rnn_maf,
#     "SA-MAF": sa_maf,
# }

# def get_arch(arch_name):
#     """ Return architecture given arch name """
#     arch_name = arch_name.upper()
#     if arch_name not in arch_dict:
#         raise KeyError(
#                 f"Unknown arch name \"{arch_name}\"."\
#                 f"Available archs are: {str(arch_dict.keys())}")
#     return arch_dict[arch_name]
