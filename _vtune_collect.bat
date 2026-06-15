@echo off
set VTUNE="D:\Program Files (x86)\Intel\oneAPI\vtune\2026.2\bin64\vtune.exe"
set PYTHON=%~dp0.venv_web313\Scripts\python.exe
set SCRIPT=%~dp0_vtune_cache_measure.py
set OUTDIR=%~dp0output\vtune_results

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo === VTune Cache Collection (Admin) ===
echo Collecting sklearn (5000 samples)...
%VTUNE% -collect uarch-exploration -result-dir "%OUTDIR%\sklearn_5k" -- %PYTHON% "%SCRIPT%" --algo sklearn --samples 5000 2>"%OUTDIR%\sklearn_5k_stderr.txt"

echo Collecting histogram (5000 samples)...
%VTUNE% -collect uarch-exploration -result-dir "%OUTDIR%\histogram_5k" -- %PYTHON% "%SCRIPT%" --algo histogram --samples 5000 2>"%OUTDIR%\histogram_5k_stderr.txt"

echo Collecting sklearn (20000 samples)...
%VTUNE% -collect uarch-exploration -result-dir "%OUTDIR%\sklearn_20k" -- %PYTHON% "%SCRIPT%" --algo sklearn --samples 20000 2>"%OUTDIR%\sklearn_20k_stderr.txt"

echo Collecting histogram (20000 samples)...
%VTUNE% -collect uarch-exploration -result-dir "%OUTDIR%\histogram_20k" -- %PYTHON% "%SCRIPT%" --algo histogram --samples 20000 2>"%OUTDIR%\histogram_20k_stderr.txt"

echo === Generating summary reports ===
%VTUNE% -report summary -result-dir "%OUTDIR%\sklearn_5k" -format csv -report-output "%OUTDIR%\sklearn_5k_summary.csv" 2>nul
%VTUNE% -report summary -result-dir "%OUTDIR%\histogram_5k" -format csv -report-output "%OUTDIR%\histogram_5k_summary.csv" 2>nul
%VTUNE% -report summary -result-dir "%OUTDIR%\sklearn_20k" -format csv -report-output "%OUTDIR%\sklearn_20k_summary.csv" 2>nul
%VTUNE% -report summary -result-dir "%OUTDIR%\histogram_20k" -format csv -report-output "%OUTDIR%\histogram_20k_summary.csv" 2>nul

echo DONE > "%OUTDIR%\collection_complete.flag"
echo === Collection complete ===
