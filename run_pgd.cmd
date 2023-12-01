@ECHO OFF
@REM run PGD attack

REM allow Linf = 10% of input vrng
REM acc: 3.960396%
REM asr: 96.039604%
REM pcr: 78.217822%
python run_pgd.py --steps 10 --eps pi/10 --alpha pi/100

REM allow Linf = 3.137% of input vrng (analogue to PGD over images)
REM acc: 56.435644%
REM asr: 43.564356%
REM pcr: 25.742574%
python run_pgd.py --steps 10 --eps pi*8/255 --alpha pi*1/255

REM allow Linf = 3.137% of input vrng (analogue to PGD over images), larger alpha step
REM acc: 47.524752%
REM asr: 52.475248%
REM pcr: 34.653465%
python run_pgd.py --steps 10 --eps pi*8/255 --alpha pi*2/255

REM allow Linf = 1% of input vrng
REM acc: 76.237624%
REM asr: 23.762376%
REM pcr: 5.940594%
python run_pgd.py --steps 10 --eps pi/100 --alpha pi/1000
