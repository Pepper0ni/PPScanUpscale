#mkdir -p ./temp;
mkdir -p ./output;

bash ./Scripts/subScripts/BNSplitAndProcess.bash $1

bash ./Scripts/subScripts/BNProcessScript.bash ./postprocess/ '-f True'

#rm -r ./temp

