from src.features.schema import *
from src.features.feature import *
from src.features.pca import *

# common QNN training features
feats = [
  'Title',            # [0, 4]
  'cnt(Name_lst)',    # [1, 6]
  'cnt(Name_fst)',    # [1, 13]
  'Name_ex',          # [0, 1]
  'bin(Age)',         # [0, 4]
  'cat(Sex)',         # [0, 1]
  'Parch',            # [0, 6]
  'Family_add',       # [0, 10]
  'Family_sub-min',   # [0, 11]
  'Family_sub_abs',   # [0, 6]
  #'Family_mul',       # [0, 16]
  'Pclass-1',         # [0, 2]
  'bin(Fare)',        # [0, 5]
  'log(Fare)',        # [1.3894144, 4.849553]
  'bin(log(Fare))',   # [0, 4]
  'cat(Cabin_pf)',    # [0, 8]
  'cat(Ticket_pf)',   # [0, 11]
]
