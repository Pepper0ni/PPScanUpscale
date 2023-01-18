#mkdir -p ./temp;
bash ./Scripts/subScripts/BNSplitAndProcess.bash $1

bash ./Scripts/ubScripts/BNSingleCardProcessScript.bash $1 '-f True'
