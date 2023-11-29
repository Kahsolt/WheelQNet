@ECHO OFF
@REM batch run_vqnet experiments on baseline `hea_amp`

:abla_B
python run_vqnet.py -M hea_amp -B 16  --name hea_amp_B=16
python run_vqnet.py -M hea_amp -B 32  --name hea_amp_B=32
python run_vqnet.py -M hea_amp -B 64  --name hea_amp_B=64
python run_vqnet.py -M hea_amp -B 96  --name hea_amp_B=96
python run_vqnet.py -M hea_amp -B 128 --name hea_amp_B=128

REM Acc: 81.188119%
python run_vqnet.py -L log\hea_amp_B=16
REM Acc: 82.178218%
python run_vqnet.py -L log\hea_amp_B=32
REM Acc: 81.188119%
python run_vqnet.py -L log\hea_amp_B=64
REM Acc: 80.198020%
python run_vqnet.py -L log\hea_amp_B=96
REM Acc: 79.207921%
python run_vqnet.py -L log\hea_amp_B=128


:abla_O
python run_vqnet.py -M hea_amp -O SGD      --name hea_amp_O=SGD
python run_vqnet.py -M hea_amp -O Adagrad  --name hea_amp_O=Adagrad
python run_vqnet.py -M hea_amp -O Adadelta --name hea_amp_O=Adadelta
python run_vqnet.py -M hea_amp -O RMSProp  --name hea_amp_O=RMSProp
python run_vqnet.py -M hea_amp -O Adam     --name hea_amp_O=Adam
python run_vqnet.py -M hea_amp -O Adamax   --name hea_amp_O=Adamax

REM Acc: 81.188119%
python run_vqnet.py -L log\hea_amp_O=SGD
REM Acc: 82.178218%
python run_vqnet.py -L log\hea_amp_O=Adagrad
REM Acc: 65.346535%
python run_vqnet.py -L log\hea_amp_O=Adadelta
REM Acc: 78.217822%
python run_vqnet.py -L log\hea_amp_O=RMSProp
REM Acc: 82.178218%
python run_vqnet.py -L log\hea_amp_O=Adam
REM Acc: 64.356436%
python run_vqnet.py -L log\hea_amp_O=Adamax


:abla_lr
python run_vqnet.py -M hea_amp --lr 0.1   --name hea_amp_lr=0.1
python run_vqnet.py -M hea_amp --lr 0.01  --name hea_amp_lr=0.01
python run_vqnet.py -M hea_amp --lr 0.001 --name hea_amp_lr=0.001

REM Acc: 81.188119%
python run_vqnet.py -L log\hea_amp_lr=0.1
REM Acc: 75.247525%
python run_vqnet.py -L log\hea_amp_lr=0.01
REM Acc: 68.316832%
python run_vqnet.py -L log\hea_amp_lr=0.001


:abla_D
python run_vqnet.py -M hea_amp -D 1 --name hea_amp_D=1
python run_vqnet.py -M hea_amp -D 2 --name hea_amp_D=2
python run_vqnet.py -M hea_amp -D 3 --name hea_amp_D=3
python run_vqnet.py -M hea_amp -D 4 --name hea_amp_D=4

REM Acc: 71.287129%
python run_vqnet.py -L log\hea_amp_D=1
REM Acc: 75.247525%
python run_vqnet.py -L log\hea_amp_D=2
REM Acc: 81.188119%
python run_vqnet.py -L log\hea_amp_D=3
REM Acc: 78.217822%
python run_vqnet.py -L log\hea_amp_D=4


:abla_gates
python run_vqnet.py -M hea_amp --hea_rots RX                      --name hea_amp_rots=X
python run_vqnet.py -M hea_amp --hea_rots RY                      --name hea_amp_rots=Y
python run_vqnet.py -M hea_amp --hea_rots RX,RY                   --name hea_amp_rots=XY
python run_vqnet.py -M hea_amp --hea_rots RX,RY,RZ                --name hea_amp_rots=XYZ
python run_vqnet.py -M hea_amp --hea_rots RY,RZ,RY                --name hea_amp_rots=YZY
python run_vqnet.py -M hea_amp --hea_rots RX       --hea_entgl CZ --name hea_amp_rots=X_entgl=Z
python run_vqnet.py -M hea_amp --hea_rots RX,RY    --hea_entgl CZ --name hea_amp_rots=XY_entgl=Z
python run_vqnet.py -M hea_amp --hea_rots RX,RY,RZ --hea_entgl CZ --name hea_amp_rots=XYZ_entgl=Z
python run_vqnet.py -M hea_amp --hea_rots RY,RZ,RY --hea_entgl CZ --name hea_amp_rots=YZY_entgl=Z

REM Acc: 64.356436%
python run_vqnet.py -L log\hea_amp_rots=X
REM Acc: 81.188119%
python run_vqnet.py -L log\hea_amp_rots=Y
python run_vqnet.py -L log\hea_amp_rots=XY
python run_vqnet.py -L log\hea_amp_rots=XYZ
python run_vqnet.py -L log\hea_amp_rots=YZY
REM Acc: 58.415842%
python run_vqnet.py -L log\hea_amp_rots=X_entgl=Z
REM Acc: 78.217822%
python run_vqnet.py -L log\hea_amp_rots=XY_entgl=Z
python run_vqnet.py -L log\hea_amp_rots=XYZ_entgl=Z
python run_vqnet.py -L log\hea_amp_rots=YZY_entgl=Z


:abla_rules
python run_vqnet.py -M hea_amp --hea_rots RX,RY    --hea_entgl_rule all --name hea_amp_rots=XY_rule=all
python run_vqnet.py -M hea_amp --hea_rots RX,RZ,RX --hea_entgl_rule all --name hea_amp_rots=XZX_rule=all
python run_vqnet.py -M hea_amp --hea_rots RY,RZ,RY --hea_entgl_rule all --name hea_amp_rots=YZY_rule=all

REM Acc: 80.198020%
python run_vqnet.py -L log\hea_amp_rots=XY_rule=all
REM Acc: 64.356436%
python run_vqnet.py -L log\hea_amp_rots=XZX_rule=all
REM Acc: 81.188119%
python run_vqnet.py -L log\hea_amp_rots=YZY_rule=all
