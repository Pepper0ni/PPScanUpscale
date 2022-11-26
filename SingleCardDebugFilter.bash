#mkdir -p ./temp;
bash ./SplitAndProcess.bash $1

bash ./SingleCardProcessScript.bash $1 '-f True -d True -s True'

