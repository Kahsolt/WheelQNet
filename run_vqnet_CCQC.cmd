@ECHO OFF
@REM batch run_vqnet experiments on baseline `ccqc`

:abla_B
python run_vqnet.py -M ccqc -B 16  --name ccqc_B=16
python run_vqnet.py -M ccqc -B 32  --name ccqc_B=32
python run_vqnet.py -M ccqc -B 64  --name ccqc_B=64
python run_vqnet.py -M ccqc -B 96  --name ccqc_B=96
python run_vqnet.py -M ccqc -B 128 --name ccqc_B=128

REM Acc: 78.217822%
python run_vqnet.py -L log\ccqc_B=16
REM Acc: 79.207921%
python run_vqnet.py -L log\ccqc_B=32
REM Acc: 79.207921%
python run_vqnet.py -L log\ccqc_B=64
REM Acc: 81.188119%
python run_vqnet.py -L log\ccqc_B=96
REM Acc: 80.198020%
python run_vqnet.py -L log\ccqc_B=128


:abla_O
python run_vqnet.py -M ccqc -O SGD      --name ccqc_O=SGD
python run_vqnet.py -M ccqc -O Adagrad  --name ccqc_O=Adagrad
python run_vqnet.py -M ccqc -O Adadelta --name ccqc_O=Adadelta
python run_vqnet.py -M ccqc -O RMSProp  --name ccqc_O=RMSProp
python run_vqnet.py -M ccqc -O Adam     --name ccqc_O=Adam
python run_vqnet.py -M ccqc -O Adamax   --name ccqc_O=Adamax

REM Acc: 79.207921%
python run_vqnet.py -L log\ccqc_O=SGD
REM Acc: 82.178218%
python run_vqnet.py -L log\ccqc_O=Adagrad
REM Acc: 77.227723%
python run_vqnet.py -L log\ccqc_O=Adadelta
REM Acc: 82.178218%
python run_vqnet.py -L log\ccqc_O=RMSProp
REM Acc: 80.198020%
python run_vqnet.py -L log\ccqc_O=Adam
REM Acc: 81.188119%
python run_vqnet.py -L log\ccqc_O=Adamax


:abla_lr
python run_vqnet.py -M ccqc --lr 0.1   --name ccqc_lr=0.1
python run_vqnet.py -M ccqc --lr 0.01  --name ccqc_lr=0.01
python run_vqnet.py -M ccqc --lr 0.001 --name ccqc_lr=0.001

REM Acc: 79.207921%
python run_vqnet.py -L log\ccqc_lr=0.1
REM Acc: 76.237624%
python run_vqnet.py -L log\ccqc_lr=0.01
REM Acc: 76.237624%
python run_vqnet.py -L log\ccqc_lr=0.001
