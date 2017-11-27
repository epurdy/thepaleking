"""(Unfinished) A T2T modality that imports all the useful information from
spacy, including word embeddings, and concatenates it into a nice tensor"""
import tensorflow as tf
from tensor2tensor.utils import modality
from tensor2tensor.utils import registry

from thepaleking.utils.spacy_borg import SpacyBorg

@registry.register_symbol_modality('spacy_text')
class SpacyText(modality.Modality):
    def __init__(self):
        self.spacy = SpacyBorg()
        # need to build the embeddings matrix
        self.embeddings = ...
        
    @property
    def top_dimensionality(self):
        return self.spacy.embed_dim

    def bottom(self, inputs):
        """The part of the network that loads input into the
        network. Currently just word embeddings.
        """
        # go from [batch, seqlen, 1, 1] to [batch, seqlen]
        inputs = tf.squeeze(inputs, axis=[2, 3])
        rv = tf.gather(self.embeddings, inputs)

        # go from [batch, seqlen, dim] to [batch, seqlen, 1, dim]
        rv = tf.expand_dims(rv, 2)

        return rv

    def targets_bottom(self, targets):
        """This should never be run as things are, so add an assert False to
        make sure that it never is.
        """
        with tf.control_dependencies(tf.Assert(False)):
            rv = tf.identity(targets)

        return rv

    def top(self, body_output, _):
        """Generate logits"""
        logits = tf.matmul(body_output, embeddings, transpose_b=True)

        return logits
