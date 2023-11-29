# OriginQ-2023-VQNet-Titanic-Problem

    第一届量子信息技术与应用创新大赛 本源量子VQNet量子机器学习大赛赛道

----

contest page: [https://contest.originqc.com.cn/contest/32/contest:introduction](https://contest.originqc.com.cn/contest/32/contest:introduction)  
team name: 做好坠机准备  


### quick start

⚪ install

- `conda create -n vq python==3.8`
- `conda activare vq`
- `pip install -r requirements.txt`

⚪ run eval (on pretrained)

- `python -m src.preprocess -f`
- `python -m src.eval -L <logdir>`

⚪ run train (reproduce the submission)

- `python -m src.preprocess -f`
- `python -m src.train`

⚪ development

- `pip install -r requirements_dev.txt`
- `python preprocess.py -f`
- `python run_sklearn.py`
- `python run_vqnet.py`
  - see exmaples in `run_vqnet_*.cmd`


#### refenrence

⚪ Q framework

- QPanda: [https://qpanda-tutorial.readthedocs.io/zh/latest/](https://qpanda-tutorial.readthedocs.io/zh/latest/)
- PyQPanda: [https://pyqpanda-toturial.readthedocs.io/zh/latest/index.html](https://pyqpanda-toturial.readthedocs.io/zh/latest/index.html)
- VQNet: [https://vqnet20-tutorial.readthedocs.io/en/latest/](https://vqnet20-tutorial.readthedocs.io/en/latest/)

⚪ problem & data

- kaggle page: [https://www.kaggle.com/competitions/titanic/overview](https://www.kaggle.com/competitions/titanic/overview)
- solution guide: [https://towardsdatascience.com/a-beginners-guide-to-kaggle-s-titanic-problem-3193cb56f6ca](https://towardsdatascience.com/a-beginners-guide-to-kaggle-s-titanic-problem-3193cb56f6ca)

----
by Armit
2023/10/27
