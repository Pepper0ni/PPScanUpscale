#mkdir -p ./temp;
mkdir -p ./output;

bash ./SplitAndProcess.bash $1

bash ./RunProcessScript.bash ./postprocess/ '-f True'

#rm -r ./temp

