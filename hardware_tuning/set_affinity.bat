@echo off
set AFFINITY_MASK=0x55
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
if "%~1"=="" ( echo Usage: set_affinity.bat ^<script.py^> & exit /b 1 )
start /affinity %AFFINITY_MASK% python %*
