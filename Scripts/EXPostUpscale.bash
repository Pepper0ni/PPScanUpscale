#mkdir -p ./temp;
mkdir -p ./output;

bash ./Scripts/subScripts/EXPostProcess.bash $1

bash ./Scripts/subScripts/EXProcessScript.bash ./postprocess/ ./clean/

#rm -r ./temp

