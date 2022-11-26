#mkdir -p ./temp;

bash ./SplitAndProcess.bash ./upscaled

bash ./RunProcessScript.bash ./postprocess/ '-ma True'

#rm -r ./temp
