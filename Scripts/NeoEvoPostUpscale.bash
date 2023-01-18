#mkdir -p ./temp;
mkdir -p ./output;

bash ./Scripts/subScripts/BNSplitAndProcess.bash $1

bash ./Scripts/subScripts/NeoEvoProcessScript.bash ./postprocess/

#rm -r ./temp
