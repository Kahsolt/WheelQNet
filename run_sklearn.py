#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/28

from argparse import ArgumentParser

from tqdm import tqdm
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
from src.features.pca import *
from utils import *

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
feats_all = sorted(set(feats_cat + feats_num))
feats_sn = [
  'Ticket_no',
  'Cabin_no',
]


def apply_pca(args, X_train, X_test):
  if args.method == 'tsne':
    X_concat = pd.concat([X_train, X_test], axis=0)
    X_concat = run_pca(args, X_concat)
    X_train = X_concat[:len(X_train)]
    X_test = X_concat[len(X_train):]
  else:
    X_train, (scaler, reducer) = run_pca(args, X_train, ret_ops=True)
    if scaler: X_test = scaler.transform(X_test)
    X_test = reducer.transform(X_test)
  return X_train, X_test


def run(args):
  # train data
  df = get_data('train')
  X_train_cat = df[feats_cat]
  X_train_num = df[feats_num]
  X_train_all = df[feats_all]
  Y = df[TARGET]
  # test data
  df = get_data('test')
  X_test_cat = df[feats_cat]
  X_test_num = df[feats_num]
  X_test_all = df[feats_all]
  truth = get_truth()
  del df

  print('Features:')
  print(' ', X_train_all.columns.tolist())
  print()

  if args.pca:
    pca_args = get_pca_args()
    print(f'PCA:')
    print(f'  cat: {len(X_train_cat.columns)} => {pca_args.dim}')
    X_train_cat, X_test_cat = apply_pca(pca_args, X_train_cat, X_test_cat)
    print(f'  num: {len(X_train_num.columns)} => {pca_args.dim}')
    X_train_num, X_test_num = apply_pca(pca_args, X_train_num, X_test_num)
    print(f'  all: {len(X_train_all.columns)} => {pca_args.dim}')
    X_train_all, X_test_all = apply_pca(pca_args, X_train_all, X_test_all)

  print('Accuracy:')
  acc_list, acc_test_list = [], []
  k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
  for clf in tqdm(MODELS, desc='Models'):
    seed_everything(SEED)

    model_name = clf.__class__.__name__
    if model_name in ['MultinomialNB', 'BernoulliNB', 'CategoricalNB']:
      X_train, X_test = X_train_cat, X_test_cat
    else:
      X_train, X_test = X_train_all, X_test_all

    try:
      # train/eval with cv
      scores = cross_val_score(clf, X_train, Y, cv=k_fold, n_jobs=4, scoring='accuracy')
      acc_cv = scores.mean()
      # train with all data
      clf.fit(X_train, Y)
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


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--pca', action='store_true', help='enable dimension reduction')
  args, _ = parser.parse_known_args()
  
  run(args)
