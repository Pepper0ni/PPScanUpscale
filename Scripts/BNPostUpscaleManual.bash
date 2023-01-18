#mkdir -p ./temp;

bash ./Scripts/subScripts/BNSplitAndProcess.bash ./upscaled

bash ./Scripts/subScripts/BNProcessScript.bash ./postprocess/ '-ma True'

#rm -r ./temp
