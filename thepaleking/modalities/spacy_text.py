"""(Unfinished) A T2T modality that imports all the useful information from
spacy, including word embeddings, and concatenates it into a nice tensor"""

from tensor2tensor.utils import modality
from tensor2tensor.utils import registry

from thepaleking.utils.spacy_borg import SpacyBorg

@registry.register_symbol_modality('spacy_text')
class SpacyText(modality.Modality):
    def __init__(self):
        self.spacy = SpacyBorg()

    @property
    def top_dimensionality(self):
        return self.spacy.embed_dim
