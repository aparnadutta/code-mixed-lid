from indictrans import Transliterator

trn = Transliterator(source='eng', target='ben', decode='beamsearch')

ex_bangla_word = 'jugopojogi'

bengali = trn.transform(ex_bangla_word, k_best=5)

