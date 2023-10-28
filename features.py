#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/27

import math
from pathlib import Path
from re import compile as Regex
from collections import Counter, defaultdict

import numpy as np
from pandas import DataFrame
import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt

from utils import *
from schema import *

Stats = Dict[str, Any]
Cntr = DefaultDict[str, int]

TARGET = 'Survived'
NO_STR = '_'
NO_INT = -1

# e.g.: Hirvonen, Mrs. Alexander (Helga E Lindqvist)
R_NAME_FAMILY = Regex(r'^([^,]+),')
R_NAME_TITLE  = Regex(r', ([^\.]+\.)')
R_NAME_FIRST  = Regex(r'\. ([^\("]+)')
R_NAME_EXTRA  = Regex(r'[\("](.+)["\)]')
R_NUMBER      = Regex(r'\d+')

TITLE_MAPPING = {
  NO_STR: 0,    # others
  'Mr.': 1,
  'Mrs.': 2,
  'Miss.': 3,
  'Master.': 4,
#  'Dr.': 5,
#  'Rev.': 6,
}

SEX_MAPPING = {
  'female': 0,
  'male': 1,
}

EMBARKED_MAPPING = {  # port of Embarkation
  'S': 0,   # Southampton
  'C': 1,   # Cherbourg
  'Q': 2,   # Queenstown
}

CABIN_MAPPING = {
  NO_STR: 0,  # value missing
  'A': 1,
  'B': 2,
  'C': 3,
  'D': 4,
  'E': 5,
  'F': 6,
  'G': 7,
  'T': 8,
}

TICKET_PREFIX_MAPPING = {
  NO_STR: 0,  # others
  'PC': 1, 
  'CA': 2, 
  'A5': 3, 
  'SOTONOQ': 4, 
  'STONO': 5, 
  'WC': 6, 
  'SCPARIS': 7, 
  'A4': 8, 
  'SOC': 9, 
  'STONO2': 10, 
  'FCC': 11, 
}

def name_title(s:str) -> int:
  return TITLE_MAPPING.get(R_NAME_TITLE.findall(s)[0], 0)

def name_family(s:str) -> str:
  m: List[str] = R_NAME_FAMILY.findall(s)
  return m[0].strip() if m else NO_STR

def name_first(s:str) -> str:
  m: List[str] = R_NAME_FIRST.findall(s)
  return m[0].strip() if m else NO_STR

def name_ex(s:str) -> str:
  m: List[str] = R_NAME_EXTRA.findall(s)
  return m[0].strip().strip('"') if m else NO_STR

def age_bin(x:float) -> int:
  if x <= 16: return 0
  if x <= 26: return 1
  if x <= 36: return 2
  if x <= 58: return 3
  return 4

def ticket_pf(s:str) -> str:
  return s.split(' ')[0].upper().replace('/', '').replace('.', '') if ' ' in s else NO_STR

def ticket_no(s:str) -> int:
  if not isinstance(s, str): return NO_INT
  x = s.split(' ')[-1] if ' ' in s else s
  return int(x) if x.isdigit() else NO_INT

def fare_bin(x:float) -> int:
  if x <= 17: return 0
  if x <= 32: return 1
  if x <= 45: return 2
  if x <= 100: return 3
  if x <= 200: return 4
  return 5

def log_fare_bin(x:float) -> int:
  if x <= 2.25: return 0
  if x <= 2.9: return 1
  if x <= 3.8: return 2
  if x <= 4.6: return 3
  return 4

def cabin_pf(s:str) -> str:
  if not isinstance(s, str): return NO_STR
  if s.startswith('F '): s = s[2:]    # fix case: 'F E69'/'F G63'/'F G73'
  return s[0] if s[0] in CABIN_MAPPING else NO_STR

def cabin_no(s:str) -> int:
  if not isinstance(s, str): return NO_INT
  m = R_NUMBER.findall(s)
  if not m: return NO_INT
  nums = [int(x) for x in m]    # multi-seats compatible
  return round(mean(nums))


def lookup_namelist(cntr:Cntr, s:str) -> int:
  if s is NO_STR: return 0
  frqs = []
  for seg in s.lower().replace('-', ' ').split(' '):
    if seg in cntr:
      frqs.append(cntr[seg])
  return round(mean(frqs))

def load_namelists(dp:Path) -> Dict[str, Cntr]:
  namelists = { }
  for x in ['dead', 'live']:
    for y in ['lst', 'fst']:
      name = f'{x}_{y}'
      namelists[name] = load_json(dp / f'{name}.json')
  return namelists

def extract_namelists(df:DataFrame, dp:Path):
  if TARGET not in list(df.columns): return

  def count_segs(df:DataFrame, last_cntr:Cntr, first_cntr:Cntr) -> Tuple[Set[str], Set[str]]:
    def parse_and_put(fullname:str, parser:Callable, cntr:Cntr):
      partname: str = parser(fullname)
      if partname is NO_STR: return
      for seg in partname.replace('-', ' ').split(' '):
        if len(seg) <= 2: continue
        cntr[seg.lower()] += 1

    for fullname in df['Name']:
      parse_and_put(fullname, name_family, last_cntr)
      parse_and_put(fullname, name_first, first_cntr)

  live = df[df[TARGET] == 1]
  live_lst, live_fst = defaultdict(int), defaultdict(int)
  count_segs(live, live_lst, live_fst)
  dead = df[df[TARGET] == 0]
  dead_lst, dead_fst = defaultdict(int), defaultdict(int)
  count_segs(dead, dead_lst, dead_fst)

  for x in ['dead', 'live']:
    for y in ['lst', 'fst']:
      name = f'{x}_{y}'
      cntr: Cntr = locals()[name]
      cnt_val = sorted([(cnt, val) for val, cnt in cntr.items()], reverse=True)
      data = {val: cnt for cnt, val in cnt_val}
      save_json(data, dp / f'{name}.json')


def extract_features(df:DataFrame, namelists:Dict={}, stats:Stats={}) -> Tuple[DataFrame, Stats]:
  '''
    - ref: https://github.com/ashishpatel26/Titanic-Machine-Learning-from-Disaster#4-feature-engineering
    - feature names: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
  '''

  df['Pclass-1'] = df['Pclass'] - 1   # offset to 0~2

  df['Title'] = df['Name'].map(name_title)
  df['Name_ex'] = df['Name'].map(lambda e: int(name_ex(e) is NO_STR))
  df['Name_lst'] = df['Name'].map(name_family)
  df['Name_fst'] = df['Name'].map(name_first)
  names_lst = stats.get('names_lst', Counter(df['Name_lst']))
  names_fst = stats.get('names_fst', Counter(df['Name_fst']))
  df['cnt(Name_lst)'] = df['Name_lst'].map(lambda e: names_lst.get(e, 0))
  df['cnt(Name_fst)'] = df['Name_fst'].map(lambda e: names_fst.get(e, 0))
  df['freq(Name_live_lst)'] = df['Name_lst'].map(lambda e: lookup_namelist(namelists['live_lst'], e))
  df['freq(Name_live_fst)'] = df['Name_fst'].map(lambda e: lookup_namelist(namelists['live_fst'], e))
  df['freq(Name_dead_lst)'] = df['Name_lst'].map(lambda e: lookup_namelist(namelists['dead_lst'], e))
  df['freq(Name_dead_fst)'] = df['Name_fst'].map(lambda e: lookup_namelist(namelists['dead_fst'], e))

  df['cat(Sex)'] = df['Sex'].map(SEX_MAPPING)

  age_fill = stats.get('age_fill', df.groupby('Title')['Age'].transform('median'))
  df['Age'].fillna(age_fill, inplace=True)
  df['log(Age)'] = df['Age'].map(lambda e: math.log(e + 1))   # offset by 1
  df['bin(Age)'] = df['Age'].map(age_bin)

  df['Family_add'] = df['SibSp'] + df['Parch']
  df['Family_sub'] = df['SibSp'] - df['Parch']
  family_sub_min = stats.get('family_sub_min', df['Family_sub'].min())
  df['Family_sub-min'] = (df['Family_sub'] - family_sub_min).clip(lower=0, upper=11)  # MAGIC: just do
  df['Family_sub_abs'] = np.abs(df['Family_sub'])
  df['Family_mul'] = df['SibSp'] * df['Parch']

  df['cat(Ticket_pf)'] = df['Ticket'].map(ticket_pf).map(lambda e: TICKET_PREFIX_MAPPING.get(e, 0))
  df['Ticket_no'] = df['Ticket'].map(ticket_no)

  df['Fare'].replace(0.0, np.nan, inplace= True)    # remove outlier
  fare_fill = stats.get('fare_fill', df.groupby('Pclass')['Fare'].transform('median'))
  df['Fare'].fillna(fare_fill, inplace=True)
  df['bin(Fare)'] = df['Fare'].map(fare_bin)
  df['log(Fare)'] = df['Fare'].map(math.log)
  df['bin(log(Fare))'] = df['log(Fare)'].map(log_fare_bin)

  df['cat(Cabin_pf)'] = df['Cabin'].map(cabin_pf).map(CABIN_MAPPING)
  df['Cabin_no'] = df['Cabin'].map(cabin_no)

  df['Embarked'].fillna('S', inplace=True)
  df['cat(Embarked)'] = df['Embarked'].map(EMBARKED_MAPPING)

  del df['Pclass']
  del df['Name']
  del df['Sex']
  del df['Ticket']
  del df['Cabin']
  del df['Embarked']

  del df['Name_lst']
  del df['Name_fst']

  df = df.convert_dtypes()
  df.info(verbose=True)

  stats = {
    'names_lst': names_lst,
    'names_fst': names_fst,
    'family_sub_min': family_sub_min,
    'age_fill':  age_fill,
    'fare_fill': fare_fill,
  }
  return df, stats


def extract_hists(df:DataFrame, dp:Path):
  if TARGET in list(df.columns):
    live = df[df[TARGET] == 1]
    dead = df[df[TARGET] == 0]

    for feat in list(df.columns):
      if feat == TARGET: continue

      n_vals = len(set(df[feat]))
      n_bins = min(n_vals, 50)

      plt.clf()
      plt.subplot(121) ; plt.title('live') ; plt.hist(live[feat], bins=n_bins)
      plt.subplot(122) ; plt.title('dead') ; plt.hist(dead[feat], bins=n_bins)
      plt.suptitle(feat)
      fp = dp / f'{feat}.png'
      print(f'>> save to {fp}')
      plt.savefig(fp, dpi=600)
  else:
    for feat in list(df.columns):
      n_vals = len(set(df[feat]))
      n_bins = min(n_vals, 50)

      plt.clf()
      plt.hist(df[feat], bins=n_bins)
      plt.suptitle(feat)
      fp = dp / f'{feat}.png'
      print(f'>> save to {fp}')
      plt.savefig(fp, dpi=600)
