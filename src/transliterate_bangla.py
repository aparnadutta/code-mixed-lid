from indictrans import Transliterator

trn = Transliterator(source='eng', target='ben', decode='beamsearch')

# TODO
#   Experiment with normalize(text)

bengali = trn.transform("jugopojogi", k_best=5)

# print(tags)

