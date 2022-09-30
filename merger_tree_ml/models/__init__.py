
from . import bijectors
from . import maf
from . import gru

arch_dict = {
    "maf": maf
    "gru": gru
}

def get_arch(arch_name):
    """ Return architecture given arch name """
    arch_name = arch_name.lower()
    if arch_name not in arch_dict:
        raise ValueError(f"arch {} not in dictionary")
    return arch_dict[arch_name]
