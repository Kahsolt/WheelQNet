#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/27

from re import compile as Regex
from dataclasses import dataclass
from typing import *

U = Union
L = Literal
R_NAME = Regex('([^,]+) ([^\.]+\.) (.+)')


@dataclass
class Schema:
  # 唯一标识，从 1 起计数
  PassengerId: int
  # 生还，二分类目标
  Survived: U[L[0], L[1]]
  # 舱位等级
  Pclass: U[L[1], L[2], L[3]]
  # 姓名，形如 "名, Mr/Mrs/Miss/Dr/Major/Master/Capt/Rev/Col./Don. 姓氏"
  Name: str
  # 性别
  Sex: U[L['male'], L['female']]
  # 年龄，会有 x.5 岁的情况
  Age: float
  # 同行的兄弟姐妹数量
  SibSp: int
  # 同行的父母孩子数量
  Parch: int
  # 票号
  Ticket: int
  # 票价
  Fare: float
  # 船舱号 (有空值、多值)
  Cabin: str
  # 登船港口
  Embarked: U[L['S'], L['C'], L['Q']]

