#mkdir -p ./temp;
mkdir -p ./output;

bash ./Scripts/subScripts/EXPostProcess.bash $1

bash ./Scripts/subScripts/deltaProcessScript2.bash ./postprocess/

#rm -r ./temp

