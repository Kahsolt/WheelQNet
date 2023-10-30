#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/28

from src.features import TARGET
from src.train import *
from src.eval import *
from utils import *

seed_everything(SEED)

feats_id = 'PassengerId'
feats_tgt = 'Survived'
feats_cat_or_num = [
  'cnt(Name_lst)',          # ok
  'cnt(Name_fst)',          # ok
  #'freq(Name_live_lst)',   # overfit
  #'freq(Name_live_fst)',   # overfit
  #'freq(Name_dead_lst)',   # overfit
  #'freq(Name_dead_fst)',   # overfit
  'bin(Age)',
  #'SibSp',
  'Parch',
  'Family_add',
  'Family_sub-min',
  'Family_sub_abs',
  'Family_mul',
  'Pclass-1',         # good
  'bin(Fare)',
  'bin(log(Fare))',
  'cat(Cabin_pf)',
  'cat(Ticket_pf)',
]
feats_cat = [
  'Title',            # good
  'Name_ex',          # good
  'cat(Sex)',         # good
  #'cat(Embarked)',   # no effect
] + feats_cat_or_num
feats_num = [
  #'Age',         # bad
  #'log(Age)',    # bad
  #'Fare',        # bad
  'log(Fare)',    # good
  #'Family_sub',  # bad
] + feats_cat_or_num
feats_sn = [
  'Ticket_no',
  'Cabin_no',
]

# train data
df = get_data('train')
X_cat = df[feats_cat]
X_num = df[feats_num]
X_all = df[feats_cat + feats_num]
Y = df[TARGET]
# test data
df = get_data('test')
X_test_cat = df[feats_cat]
X_test_num = df[feats_num]
X_test_all = df[feats_cat + feats_num]
truth = get_truth()
del df

print('Features:')
print(' ', X_all.columns.tolist())
print()

# TODO: train & test
