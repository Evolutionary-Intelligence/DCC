from pypoplib.mmes import MMES  # mainly for strong (ill-conditioned) variable dependency
from pypoplib.cocma import COCMA  # mainly for weak (sparse) variable dependency
from pypoplib.d_es import DES  # for distributed computing


class DCC(DES):
    """DCC: Distributed Cooperative Coevolution (powered by LM-CMA/CMA-ES under the recently proposed
        Multi-Level Learning framework).

        This class is only a wrapper for `DCC`. See `DES` in the folder *pypoplib* for details.
    """
    def __init__(self, problem, options):
        DES.__init__(self, problem, options)
        self.sub_optimizer = [MMES, COCMA]  # `MMES` is the latest version/variant of LM-CMA
