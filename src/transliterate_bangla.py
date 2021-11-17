from indictrans import Transliterator
from nltk import pos_tag as tagger

trn = Transliterator(source='eng', target='ben', decode='beamsearch')

# TODO
#   Experiment with normalize(text)

bengali = trn.transform("jugopojogi", k_best=5)

tags = tagger("yes exactly i don' t know what she meant either".split(" "))
# print(tags)

