@ECHO OFF
@REM batch run_vqnet experiments on modifed baseline `ccqcq`

:abla_B
python run_vqnet.py -M ccqcq -B 16  --name ccqcq_B=16
python run_vqnet.py -M ccqcq -B 32  --name ccqcq_B=32
python run_vqnet.py -M ccqcq -B 64  --name ccqcq_B=64
python run_vqnet.py -M ccqcq -B 96  --name ccqcq_B=96
python run_vqnet.py -M ccqcq -B 128 --name ccqcq_B=128

REM Acc: 81.188119%
python run_vqnet.py -L log\ccqcq_B=16
REM Acc: 79.207921%
python run_vqnet.py -L log\ccqcq_B=32
REM Acc: 80.198020%
python run_vqnet.py -L log\ccqcq_B=64
REM Acc: 81.188119%
python run_vqnet.py -L log\ccqcq_B=96
REM Acc: 80.198020%
python run_vqnet.py -L log\ccqcq_B=128


:abla_O
python run_vqnet.py -M ccqcq -O SGD      --name ccqcq_O=SGD
python run_vqnet.py -M ccqcq -O Adagrad  --name ccqcq_O=Adagrad
python run_vqnet.py -M ccqcq -O Adadelta --name ccqcq_O=Adadelta
python run_vqnet.py -M ccqcq -O RMSProp  --name ccqcq_O=RMSProp
python run_vqnet.py -M ccqcq -O Adam     --name ccqcq_O=Adam
python run_vqnet.py -M ccqcq -O Adamax   --name ccqcq_O=Adamax

REM Acc: 80.198020%
python run_vqnet.py -L log\ccqcq_O=SGD
REM Acc: 77.227723%
python run_vqnet.py -L log\ccqcq_O=Adagrad
REM Acc: 64.356436%
python run_vqnet.py -L log\ccqcq_O=Adadelta
REM Acc: 79.207921%
python run_vqnet.py -L log\ccqcq_O=RMSProp
REM Acc: 80.198020%
python run_vqnet.py -L log\ccqcq_O=Adam
REM Acc: 64.356436%
python run_vqnet.py -L log\ccqcq_O=Adamax


:abla_lr
python run_vqnet.py -M ccqcq --lr 0.1   --name ccqcq_lr=0.1
python run_vqnet.py -M ccqcq --lr 0.01  --name ccqcq_lr=0.01
python run_vqnet.py -M ccqcq --lr 0.001 --name ccqcq_lr=0.001

REM Acc: 80.198020%
python run_vqnet.py -L log\ccqcq_lr=0.1
REM Acc: 75.247525%
python run_vqnet.py -L log\ccqcq_lr=0.01
REM Acc: 64.356436%
python run_vqnet.py -L log\ccqcq_lr=0.001
