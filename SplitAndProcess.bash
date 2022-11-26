mkdir -p ./postprocess;
mkdir -p ./light/;
mkdir -p ./dark/;
mkdir -p ./balanced/;

python3 ./splitByValue.py -i $1

for f in $(find ./light -name "*.png" -type f); do
 bash ./PostProcessNormal.bash $f
done

for f in $(find ./balanced -name "*.png" -type f); do
 bash ./PostProcessGentle.bash $f
done

for f in $(find ./dark -name "*.png" -type f); do
 bash ./PostProcessNoGamma.bash $f
done

rm -r ./light/
rm -r ./dark/
rm -r ./balanced/
