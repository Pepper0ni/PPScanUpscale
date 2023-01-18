#mkdir -p ./temp;
mkdir -p ./output;

bash ./Scripts/subScripts/SplitAndProcess.bash $1

bash ./Scripts/subScripts/BNProcessScript.bash ./postprocess/

#rm -r ./temp

