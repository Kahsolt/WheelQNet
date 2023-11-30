@ECHO OFF
@REM batch run_vqnet experiments on baseline `hea_amp`

SET MODEL=hea_amp

:abla_B
python run_vqnet.py -M %MODEL% -B 16  --name %MODEL%_B=16
python run_vqnet.py -M %MODEL% -B 32  --name %MODEL%_B=32
python run_vqnet.py -M %MODEL% -B 64  --name %MODEL%_B=64
python run_vqnet.py -M %MODEL% -B 96  --name %MODEL%_B=96
python run_vqnet.py -M %MODEL% -B 128 --name %MODEL%_B=128

REM Acc: 81.188119%
python run_vqnet.py -L log\%MODEL%_B=16
REM Acc: 82.178218%
python run_vqnet.py -L log\%MODEL%_B=32
REM Acc: 81.188119%
python run_vqnet.py -L log\%MODEL%_B=64
REM Acc: 80.198020%
python run_vqnet.py -L log\%MODEL%_B=96
REM Acc: 79.207921%
python run_vqnet.py -L log\%MODEL%_B=128


:abla_O
python run_vqnet.py -M %MODEL% -O SGD      --name %MODEL%_O=SGD
python run_vqnet.py -M %MODEL% -O Adagrad  --name %MODEL%_O=Adagrad
python run_vqnet.py -M %MODEL% -O Adadelta --name %MODEL%_O=Adadelta
python run_vqnet.py -M %MODEL% -O RMSProp  --name %MODEL%_O=RMSProp
python run_vqnet.py -M %MODEL% -O Adam     --name %MODEL%_O=Adam
python run_vqnet.py -M %MODEL% -O Adamax   --name %MODEL%_O=Adamax

REM Acc: 81.188119%
python run_vqnet.py -L log\%MODEL%_O=SGD
REM Acc: 82.178218%
python run_vqnet.py -L log\%MODEL%_O=Adagrad
REM Acc: 65.346535%
python run_vqnet.py -L log\%MODEL%_O=Adadelta
REM Acc: 78.217822%
python run_vqnet.py -L log\%MODEL%_O=RMSProp
REM Acc: 82.178218%
python run_vqnet.py -L log\%MODEL%_O=Adam
REM Acc: 64.356436%
python run_vqnet.py -L log\%MODEL%_O=Adamax


:abla_lr
python run_vqnet.py -M %MODEL% --lr 0.1   --name %MODEL%_lr=0.1
python run_vqnet.py -M %MODEL% --lr 0.01  --name %MODEL%_lr=0.01
python run_vqnet.py -M %MODEL% --lr 0.001 --name %MODEL%_lr=0.001

REM Acc: 81.188119%
python run_vqnet.py -L log\%MODEL%_lr=0.1
REM Acc: 75.247525%
python run_vqnet.py -L log\%MODEL%_lr=0.01
REM Acc: 68.316832%
python run_vqnet.py -L log\%MODEL%_lr=0.001


:abla_D
python run_vqnet.py -M %MODEL% -D 1 --name %MODEL%_D=1
python run_vqnet.py -M %MODEL% -D 2 --name %MODEL%_D=2
python run_vqnet.py -M %MODEL% -D 3 --name %MODEL%_D=3
python run_vqnet.py -M %MODEL% -D 4 --name %MODEL%_D=4

REM Acc: 71.287129%
python run_vqnet.py -L log\%MODEL%_D=1
REM Acc: 75.247525%
python run_vqnet.py -L log\%MODEL%_D=2
REM Acc: 81.188119%
python run_vqnet.py -L log\%MODEL%_D=3
REM Acc: 78.217822%
python run_vqnet.py -L log\%MODEL%_D=4


:abla_gates
python run_vqnet.py -M %MODEL% --hea_rots RX                      --name %MODEL%_rots=X
python run_vqnet.py -M %MODEL% --hea_rots RY                      --name %MODEL%_rots=Y
python run_vqnet.py -M %MODEL% --hea_rots RX,RY                   --name %MODEL%_rots=XY
python run_vqnet.py -M %MODEL% --hea_rots RX,RY,RZ                --name %MODEL%_rots=XYZ
python run_vqnet.py -M %MODEL% --hea_rots RY,RZ,RY                --name %MODEL%_rots=YZY
python run_vqnet.py -M %MODEL% --hea_rots RX       --hea_entgl CZ --name %MODEL%_rots=X_entgl=Z
python run_vqnet.py -M %MODEL% --hea_rots RX,RY    --hea_entgl CZ --name %MODEL%_rots=XY_entgl=Z
python run_vqnet.py -M %MODEL% --hea_rots RX,RY,RZ --hea_entgl CZ --name %MODEL%_rots=XYZ_entgl=Z
python run_vqnet.py -M %MODEL% --hea_rots RY,RZ,RY --hea_entgl CZ --name %MODEL%_rots=YZY_entgl=Z

REM Acc: 64.356436%
python run_vqnet.py -L log\%MODEL%_rots=X
REM Acc: 81.188119%
python run_vqnet.py -L log\%MODEL%_rots=Y
python run_vqnet.py -L log\%MODEL%_rots=XY
python run_vqnet.py -L log\%MODEL%_rots=XYZ
python run_vqnet.py -L log\%MODEL%_rots=YZY
REM Acc: 58.415842%
python run_vqnet.py -L log\%MODEL%_rots=X_entgl=Z
REM Acc: 78.217822%
python run_vqnet.py -L log\%MODEL%_rots=XY_entgl=Z
python run_vqnet.py -L log\%MODEL%_rots=XYZ_entgl=Z
python run_vqnet.py -L log\%MODEL%_rots=YZY_entgl=Z


:abla_rules
python run_vqnet.py -M %MODEL% --hea_rots RX,RY    --hea_entgl_rule all --name %MODEL%_rots=XY_rule=all
python run_vqnet.py -M %MODEL% --hea_rots RX,RZ,RX --hea_entgl_rule all --name %MODEL%_rots=XZX_rule=all
python run_vqnet.py -M %MODEL% --hea_rots RY,RZ,RY --hea_entgl_rule all --name %MODEL%_rots=YZY_rule=all

REM Acc: 80.198020%
python run_vqnet.py -L log\%MODEL%_rots=XY_rule=all
REM Acc: 64.356436%
python run_vqnet.py -L log\%MODEL%_rots=XZX_rule=all
REM Acc: 81.188119%
python run_vqnet.py -L log\%MODEL%_rots=YZY_rule=all
