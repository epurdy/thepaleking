# thepaleking
Exploring linguistic style transfer and machine creativity

# libraries

* tensor2tensor: for higher-level tensorflow coding (copied into the repo
  rather than installed because it is not super mature and must frequently be
  edited)

* spacy: for low- and mid-level NLP tasks. May have to move to NLTK if this
  proves to be too flaky. Note that spacy is unable to parse all of Infinite
  Jest simultaneously; it seems to be a length issue, as it parses it just fine
  if split into smaller chunks.

# code organization

* thepaleking/problems: this is where we define problems that tensor2tensor
  knows about. These should inherit from Problem or TextProblem.

* thepaleking/modalities: this is where we define the input and output layer of
  the network so that the core T2T models can be agnostic about what data
  they're dealing with.

* thepaleking/models: this is where we define our own custom models.

# setup

Currently I think that spacy requires python >=3.5

* [sudo] pip install -r requirements.txt
* [sudo] python -m spacy download en
* [sudo] python -m spacy download en_vectors_web_lg
* Add tensor2tensor to your PYTHONPATH

# quickstart

The executables t2t-datagen (data ingest), t2t-trainer (training) and
t2t-decoder (decoding) in tensor2tensor/tensor2tensor/bin are the place to
start. We should wrap these in scripts pretty soon.

* we've now wrapped data ingest in thepaleking/bin/datagen

# known issues

* given the way that we're currently doing ingest, examples from the source
  domain are associated with a particular example from the target domain.
  Hopefully, given an appropriate model, this will not be a problem.