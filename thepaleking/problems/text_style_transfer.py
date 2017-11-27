"""(Unfinished) A data-ingest class for text style transfer"""
import itertools as it

from tensor2tensor.data_generators.problem import Text2TextProblem
from tensor2tensor.utils import registry

from thepaleking.utils.spacy_borg import SpacyBorg

MAX_SENTENCE_LENGTH = 300
MAX_SPACY_TEXT_LENGTH = 1000 * 1000

@registry.register_problem
class TextStyleTransfer(Text2TextProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load spacy models
        self.spacy = SpacyBorg()

    @property
    def is_character_level(self):
        return False

    @property
    def is_2d(self):
        return True

    @property
    def num_shards(self):
        return 100

    @property
    def use_subword_tokenizer(self):
        return False

    @property
    def targeted_vocab_size(self):
        return len(self.spacy.index_to_spacy)

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK

    def generator_from_paths(self, paths):
        """This is a generator that generates examples for a single domain from a set
        of file paths that are representative of that domain.

        Parameters
        ----------
        paths : list of string
        """
        for path in paths:
            text = open(path)
            subtext = True
            while subtext:
                #subtext = text[i:i + MAX_SPACY_TEXT_LENGTH]
                subtext = text.read(MAX_SPACY_TEXT_LENGTH)
                print('Parsing', len(subtext), 'chars')
                doc = self.spacy.nlp(subtext)
                print('done parsing')
            
                for sent in doc.sents:
                    spacy_ids = [word.lower for word in sent]
                    indices = [self.spacy.spacy_to_index.get(id, 2) for id in spacy_ids]
                    indices = indices[:MAX_SENTENCE_LENGTH]
                    indices = indices + [0] * (MAX_SENTENCE_LENGTH - len(indices))

                    yield indices

    def generator(self, data_dir, tmp_dir, train):
        """This is a generator that generates paired examples, which is kind of dumb
        for this particular problem.

        Parameters
        ----------
        data_dir : str
            Directory to put ingest files in
        tmp_dir : str
            Unused, but generally a temporary directory where files can be
            stored during ingest
        train : bool
            Create training set (True) or test set (False)
        """
        if train:
            dfw_paths = ['data/train1.txt', 'data/train2.txt']
        else:
            dfw_paths = ['data/test1.txt']

        wiki_paths = ['/data/enwiki-latest-pages-articles.xml']

        # independent data generators; cycle through dfw until wiki is
        # exhausted
        dfw_gen = it.cycle(self.generator_from_paths(dfw_paths))
        wiki_gen = self.generator_from_paths(wiki_paths)
        
        for i, (sent1, sent2) in enumerate(zip(dfw_gen, wiki_gen)):
            print(i)
            yield {'source_ints': sent1,
                   'target_ints': sent2}

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = int(True)

        if self.has_inputs:
            source_vocab_size = self._encoders["inputs"].vocab_size
            p.input_modality = {
                "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
            }
        target_vocab_size = self._encoders["targets"].vocab_size
        p.target_modality = (registry.Modalities.SYMBOL + ':spacy_text', target_vocab_size)
        p.input_space_id = self.input_space_id
        p.target_space_id = self.target_space_id

    def eval_metrics(self):
        return [
            metrics.Metrics.NEG_LOG_PERPLEXITY,
        ]
