#mkdir -p ./temp;
bash ./Scripts/subScripts/BNSplitAndProcess.bash $1

bash ./Scripts/subScripts/BNSingleCardProcessScript.bash $1 '-f True -d True -s True'
