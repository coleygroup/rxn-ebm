from typing import Dict, List

from tensorflow.compat.v1.keras import backend as K

from rxnebm.proposer.MT_karpov.transformer import buildNetwork, gen_beam
from rxnebm.proposer.proposer_base import Proposer


class MTKarpovProposer(Proposer):
    """Molecular Transformer (Karpov) proposer,
    wrapping around MT_karpov.transformer.buildNetwork()"""

    def __init__(self, mt_karpov_config: Dict) -> None:
        super().__init__()
        self.mdl, self.mdl_encoder, self.mdl_decoder, self.sess = \
            self.build_model(mt_karpov_config)

    @staticmethod
    def build_model(mt_karpov_config: Dict):
        mdl, mdl_encoder, mdl_decoder = buildNetwork(mt_karpov_config["layers"],
                                                     mt_karpov_config["heads"])
        mdl.load_weights(mt_karpov_config["model_path"])
        sess = K.get_session()

        return mdl, mdl_encoder, mdl_decoder, sess

    def propose(self, input_smiles: List[str],
                rxn_types: List[str],
                topk: int = 1,
                beam_size: int = 5,
                **kwargs) -> List[Dict[str, List]]:

        results = []
        with self.sess.as_default():
            for smi in input_smiles:
                result = gen_beam(mdl_encoder=self.mdl_encoder,
                                  mdl_decoder=self.mdl_decoder,
                                  T=1.2,
                                  product=smi,
                                  beam_size=beam_size,
                                  topk=topk)
                results.append(result)

        return results
