#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/27

from dataclasses import dataclass
from typing import Union as U, Literal as L


@dataclass
class Schema:
  # 唯一标识，从 1 起计数
  PassengerId: int
  # 生还，二分类目标
  Survived: U[L[0], L[1]]
  # 姓名，形如 "姓氏, 头衔. 名字 (备注?)"
  Name: str
  # 性别
  Sex: U[L['male'], L['female']]
  # 年龄，有 x.5 岁的情况; 有 20% 缺失值
  Age: float
  # 同行的兄弟姐妹数量
  SibSp: int
  # 同行的父母孩子数量
  Parch: int
  # 票号
  Ticket: int
  # 票价 (测试集有1个缺失值)
  Fare: float
  # 舱位等级
  Pclass: U[L[1], L[2], L[3]]
  # 船舱号 (有 80% 缺失值、不规则值)
  Cabin: str
  # 登船港口 (训练集有2个缺失值)
  Embarked: U[L['S'], L['C'], L['Q']]
