@ECHO OFF
@REM batch run_vqnet experiments on our proposed `knnq`

SET MODEL=knnq

:abla_k
python run_vqnet.py -M %MODEL% --knn 3  --name %MODEL%_k=3
python run_vqnet.py -M %MODEL% --knn 5  --name %MODEL%_k=5
python run_vqnet.py -M %MODEL% --knn 7  --name %MODEL%_k=7
python run_vqnet.py -M %MODEL% --knn 10 --name %MODEL%_k=10

REM Acc: 84.158416%
python run_vqnet.py -L log\%MODEL%_k=3
REM Acc: 87.128713%
python run_vqnet.py -L log\%MODEL%_k=5
REM Acc: 86.138614%
python run_vqnet.py -L log\%MODEL%_k=7
REM Acc: 84.158416%
python run_vqnet.py -L log\%MODEL%_k=10
