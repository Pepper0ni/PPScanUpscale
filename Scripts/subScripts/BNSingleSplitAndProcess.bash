mkdir -p ./postprocess;
mkdir -p ./light/;
mkdir -p ./dark/;
mkdir -p ./balanced/;

python3 ./splitByValue.py -i $1

fb=$(basename $1)

bash ./Scripts/subScripts/BNPostProcessNormal.bash ./light/"$fb"

bash ./Scripts/subScripts/BNPostProcessGentle.bash ./balanced/"$fb"

bash ./Scripts/subScripts/BNPostProcessNoGamma.bash ./dark/"$fb"


rm -r ./light/
rm -r ./dark/
rm -r ./balanced/
