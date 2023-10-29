#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/28

from tqdm import tqdm
import pandas as pd
from sklearnex import patch_sklearn; patch_sklearn()
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score

from src.features import TARGET
from utils import *

seed_everything(SEED)

MAX_ITER = 3000

MODELS = [
  SVC  (kernel='linear',  max_iter=MAX_ITER, random_state=SEED),
  #SVC  (kernel='poly',    max_iter=MAX_ITER, random_state=SEED, degree=2),
  #SVC  (kernel='rbf',     max_iter=MAX_ITER, random_state=SEED),
  #SVC  (kernel='sigmoid', max_iter=MAX_ITER, random_state=SEED),
  NuSVC(kernel='linear',  max_iter=MAX_ITER, random_state=SEED),
  NuSVC(kernel='poly',    max_iter=MAX_ITER, random_state=SEED, degree=2),
  NuSVC(kernel='rbf',     max_iter=MAX_ITER, random_state=SEED),
  #NuSVC(kernel='sigmoid', max_iter=MAX_ITER, random_state=SEED),
  LinearSVC(penalty='l1', dual=False, max_iter=MAX_ITER, random_state=SEED),
  LinearSVC(penalty='l2', dual=False, max_iter=MAX_ITER, random_state=SEED),

  LogisticRegression(solver='lbfgs',     penalty=None, max_iter=MAX_ITER, random_state=SEED),
  LogisticRegression(solver='lbfgs',     penalty='l2', max_iter=MAX_ITER, random_state=SEED),
  LogisticRegression(solver='newton-cg', penalty=None, max_iter=MAX_ITER, random_state=SEED),
  LogisticRegression(solver='newton-cg', penalty='l2', max_iter=MAX_ITER, random_state=SEED),
  LogisticRegression(solver='liblinear', penalty='l1', max_iter=MAX_ITER, random_state=SEED),
  LogisticRegression(solver='liblinear', penalty='l2', max_iter=MAX_ITER, random_state=SEED),

  RidgeClassifier(max_iter=MAX_ITER, random_state=SEED),
  SGDClassifier  (penalty='l1',      random_state=SEED),
  SGDClassifier  (penalty='l2',      random_state=SEED),

  Perceptron(penalty='l1',                        random_state=SEED),
  Perceptron(penalty='l2',                        random_state=SEED),
  Perceptron(penalty='elasticnet', l1_ratio=0.35, random_state=SEED),

  MLPClassifier(hidden_layer_sizes=[4],    max_iter=MAX_ITER, random_state=SEED),
  MLPClassifier(hidden_layer_sizes=[8],    max_iter=MAX_ITER, random_state=SEED),
  MLPClassifier(hidden_layer_sizes=[12],   max_iter=MAX_ITER, random_state=SEED),
  MLPClassifier(hidden_layer_sizes=[8, 4], max_iter=MAX_ITER, random_state=SEED),

  KNeighborsClassifier(n_neighbors=5,  p=1),
  KNeighborsClassifier(n_neighbors=7,  p=1),
  KNeighborsClassifier(n_neighbors=10, p=1),
  KNeighborsClassifier(n_neighbors=13, p=1),
  #KNeighborsClassifier(n_neighbors=5,  p=2),
  #KNeighborsClassifier(n_neighbors=7,  p=2),
  #KNeighborsClassifier(n_neighbors=10, p=2),
  #KNeighborsClassifier(n_neighbors=13, p=2),
  #RadiusNeighborsClassifier(radius=10),

  GaussianNB(),
  MultinomialNB(),
  BernoulliNB(),
  #CategoricalNB(),

  DecisionTreeClassifier(criterion='entropy', splitter='best',   max_depth=4, random_state=SEED),
  DecisionTreeClassifier(criterion='gini',    splitter='best',   max_depth=4, random_state=SEED),
  DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=4, random_state=SEED),
  DecisionTreeClassifier(criterion='gini',    splitter='random', max_depth=4, random_state=SEED),
  ExtraTreeClassifier   (criterion='entropy', splitter='best',   max_depth=4, random_state=SEED),
  ExtraTreeClassifier   (criterion='gini',    splitter='best',   max_depth=4, random_state=SEED),
  ExtraTreeClassifier   (criterion='entropy', splitter='random', max_depth=4, random_state=SEED),
  ExtraTreeClassifier   (criterion='gini',    splitter='random', max_depth=4, random_state=SEED),

  RandomForestClassifier(n_estimators=15, criterion='entropy', max_depth=2, random_state=SEED),
  RandomForestClassifier(n_estimators=15, criterion='gini',    max_depth=2, random_state=SEED),
  ExtraTreesClassifier  (n_estimators=15, criterion='entropy', max_depth=2, random_state=SEED),
  ExtraTreesClassifier  (n_estimators=15, criterion='gini',    max_depth=2, random_state=SEED),
  BaggingClassifier     (n_estimators=10, random_state=SEED),
  BaggingClassifier     (n_estimators=15, random_state=SEED),
  BaggingClassifier     (n_estimators=20, random_state=SEED),
  AdaBoostClassifier    (n_estimators=10, random_state=SEED),
  AdaBoostClassifier    (n_estimators=15, random_state=SEED),
  AdaBoostClassifier    (n_estimators=20, random_state=SEED),
  GradientBoostingClassifier(n_estimators=10, max_features=3, max_depth=4, random_state=SEED),
  GradientBoostingClassifier(n_estimators=15, max_features=3, max_depth=4, random_state=SEED),
  GradientBoostingClassifier(n_estimators=20, max_features=3, max_depth=4, random_state=SEED),
  HistGradientBoostingClassifier(max_iter=MAX_ITER, max_depth=4, random_state=SEED),
]

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
df = pd.read_csv(get_processed_fp('train'))
X_cat = df[feats_cat]
X_num = df[feats_num]
X_all = df[feats_cat + feats_num]
Y = df[TARGET]
# test data
df = pd.read_csv(get_processed_fp('test'))
X_test_cat = df[feats_cat]
X_test_num = df[feats_num]
X_test_all = df[feats_cat + feats_num]
truth = get_truth()
del df

print('Features:')
print(' ', X_all.columns.tolist())
print()

print('Accuracy:')
acc_list, acc_test_list = [], []
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
for i, clf in enumerate(tqdm(MODELS, desc='Models')):
  model_name = clf.__class__.__name__
  if model_name in ['MultinomialNB', 'BernoulliNB', 'CategoricalNB']:
    X = X_cat
    X_test = X_test_cat
  else:
    X = X_all
    X_test = X_test_all

  try:
    # train/eval with cv
    scores = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=4, scoring='accuracy')
    acc_cv = scores.mean()
    # train with all data
    clf.fit(X, Y)
    pred = clf.predict(X_test)
    # accuray pair
    acc_test = (pred == truth).sum() / len(truth)
    print(f'  {model_name}: {acc_cv:%} / {acc_test:%}')
    acc_list.append(acc_cv)
    acc_test_list.append(acc_test)
  except:
    print(f'  {model_name}: failed')
    print_exc()

print()
print('Mean accuracy:')
print(f'  train(cv): {mean(acc_list):%}')
print(f'  test: {mean(acc_test_list):%}')
