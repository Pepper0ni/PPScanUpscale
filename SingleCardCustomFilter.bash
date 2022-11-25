#mkdir -p ./temp;
mkdir -p ./output;
mkdir -p ./postprocess;
mkdir -p ./light;
mkdir -p ./dark;
mkdir -p ./balanced;

python3 ./splitByValue.py -i $1

fn=$(basename "$1")
fd=$(dirname ${1/upscaled/dark})
fl=$(dirname ${1/upscaled/light})
fb=$(dirname ${1/upscaled/balanced})
fp=$(dirname ${1/upscaled/postprocess})
echo $1

if ! [ -f "$fp"/"$fn" ]; then
 if [ -f "$fl"/"$fn" ]; then
  gmic input "$fl"/"$fn" iain_nr_2019 2,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.08,0,0,0 afre_sharpenfft 15,1 output "$fp"/"$fn"
 elif [ -f "$fb"/"$fn" ]; then
  gmic input "$fb"/"$fn" iain_nr_2019 1,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.04,0,0,0 afre_sharpenfft 15,1 output "$fp"/"$fn"
 elif [ -f "$fd"/"$fn" ]; then
  gmic input "$fd"/"$fn" iain_nr_2019 0.05,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.04,0,0,0 afre_sharpenfft 15,1 output "$fp"/"$fn"
 fi
fi

rm -r ./light
rm -r ./dark
rm -r ./balanced

python3 ./main.py -i ./postprocess/"$fn" -o ./output/"$fn" -b 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -r 734,1024 -m ./cardmask.png -e 60,54,54,54 -f True #-c ./temp/"$fn"


