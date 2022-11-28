mkdir -p ./postprocess;
mkdir -p ./light/;
mkdir -p ./dark/;
mkdir -p ./balanced/;

python3 ./splitByValue.py -i $1

fb=$(basename $1)

bash ./PostProcessNormal.bash ./light/"$fb"

bash ./PostProcessGentle.bash ./balanced/"$fb"

bash ./PostProcessNoGamma.bash ./dark/"$fb"


rm -r ./light/
rm -r ./dark/
rm -r ./balanced/
