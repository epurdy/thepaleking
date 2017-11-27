import spacy

class SpacyBorg:
    """A Borg is a simple python trick to give you something like a
    singleton. Multiple objects can be created but they all share the same
    state.

    This Borg wraps the SpaCy NLP library so that we only load its models once.
    """
    # the state of all the borg instances
    __shared_state = {}
    def __init__(self):
        # the borg trick
        self.__dict__ = self.__shared_state

        # normal spacy stuff
        if not hasattr(self, 'nlp'):
            print('Loading spacy models...')
            # presumably better but v. slow
            #self.nlp = spacy.load('en_core_web_lg')
            self.nlp = spacy.load('en')
            self.vectors = spacy.load('en_vectors_web_lg')

            # dicts mapping indices to spacy token ids and back
            self.index_to_spacy = dict(enumerate(sorted(self.vectors.vocab.vectors.keys())))
            self.spacy_to_index = {v:k for k, v in self.index_to_spacy.items()}
            print('Done loading models.')
