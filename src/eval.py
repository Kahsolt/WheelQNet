#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/29

from src.utils import *


def eval(fp:str) -> np.ndarray:
  df = pd.read_csv(fp)
  return np.zeros(len(df))


if __name__ == "__main__":
  fp = PROCESSED_PATH / 'contest' / PROVIDER_FILES['contest']['test']
  print(eval(fp))
