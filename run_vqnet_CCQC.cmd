@ECHO OFF
@REM batch run_vqnet experiments on baseline `ccqc`

SET MODEL=ccqc

:abla_B
python run_vqnet.py -M %MODEL% -B 16  --name %MODEL%_B=16
python run_vqnet.py -M %MODEL% -B 32  --name %MODEL%_B=32
python run_vqnet.py -M %MODEL% -B 64  --name %MODEL%_B=64
python run_vqnet.py -M %MODEL% -B 96  --name %MODEL%_B=96
python run_vqnet.py -M %MODEL% -B 128 --name %MODEL%_B=128

REM Acc: 78.217822%
python run_vqnet.py -L log\%MODEL%_B=16
REM Acc: 79.207921%
python run_vqnet.py -L log\%MODEL%_B=32
REM Acc: 79.207921%
python run_vqnet.py -L log\%MODEL%_B=64
REM Acc: 81.188119%
python run_vqnet.py -L log\%MODEL%_B=96
REM Acc: 80.198020%
python run_vqnet.py -L log\%MODEL%_B=128


:abla_O
python run_vqnet.py -M %MODEL% -O SGD      --name %MODEL%_O=SGD
python run_vqnet.py -M %MODEL% -O Adagrad  --name %MODEL%_O=Adagrad
python run_vqnet.py -M %MODEL% -O Adadelta --name %MODEL%_O=Adadelta
python run_vqnet.py -M %MODEL% -O RMSProp  --name %MODEL%_O=RMSProp
python run_vqnet.py -M %MODEL% -O Adam     --name %MODEL%_O=Adam
python run_vqnet.py -M %MODEL% -O Adamax   --name %MODEL%_O=Adamax

REM Acc: 79.207921%
python run_vqnet.py -L log\%MODEL%_O=SGD
REM Acc: 82.178218%
python run_vqnet.py -L log\%MODEL%_O=Adagrad
REM Acc: 77.227723%
python run_vqnet.py -L log\%MODEL%_O=Adadelta
REM Acc: 82.178218%
python run_vqnet.py -L log\%MODEL%_O=RMSProp
REM Acc: 80.198020%
python run_vqnet.py -L log\%MODEL%_O=Adam
REM Acc: 81.188119%
python run_vqnet.py -L log\%MODEL%_O=Adamax


:abla_lr
python run_vqnet.py -M %MODEL% --lr 0.1   --name %MODEL%_lr=0.1
python run_vqnet.py -M %MODEL% --lr 0.01  --name %MODEL%_lr=0.01
python run_vqnet.py -M %MODEL% --lr 0.001 --name %MODEL%_lr=0.001

REM Acc: 79.207921%
python run_vqnet.py -L log\%MODEL%_lr=0.1
REM Acc: 76.237624%
python run_vqnet.py -L log\%MODEL%_lr=0.01
REM Acc: 76.237624%
python run_vqnet.py -L log\%MODEL%_lr=0.001
