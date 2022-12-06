#mkdir -p ./temp;
mkdir -p ./output;

bash ./SplitAndProcess.bash $1

bash ./NeoEvoProcessScript.bash ./postprocess/

#rm -r ./temp
