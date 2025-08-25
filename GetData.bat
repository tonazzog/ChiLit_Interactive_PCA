set "DESTIN=%cd%"

set "SOURCE=%cd%\..\ChiLit_Topic_Modeling"

copy "%SOURCE%\data\ChiLit_metadata.csv" "%DESTIN%\data\ChiLit_metadata.csv"
copy "%SOURCE%\data\ChiLit_Chunks_200.csv" "%DESTIN%\data\ChiLit_Chunks_200.csv"
copy "%SOURCE%\optuna_200\Optuna_ProdLDA_output.pkl" "%DESTIN%\data\Optuna_ProdLDA_output.pkl"
copy "%SOURCE%\optuna_200\ProdLDA_Topic_Labels.json" "%DESTIN%\data\ProdLDA_Topic_Labels.json"