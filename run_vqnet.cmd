@ECHO OFF
@REM run the final fair compare

python run_vqnet.py -B 32 -E 40 --overwrite -M hea_amp --name hea_amp
python run_vqnet.py -B 32 -E 40 --overwrite -M ccqc    --name ccqc
python run_vqnet.py -B 32 -E 40 --overwrite -M ccqcq   --name ccqcq
python run_vqnet.py -B 32 -E 40 --overwrite -M wheelq  --name wheelq

python run_vqnet.py -L log\hea_amp
python run_vqnet.py -L log\ccqc
python run_vqnet.py -L log\ccqcq
python run_vqnet.py -L log\wheelq
python run_vqnet.py -L log\knnq
