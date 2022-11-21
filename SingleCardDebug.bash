#mkdir -p ./temp;
mkdir -p ./output;
mkdir -p ./postprocess;

fn=$(basename "$1")
fd=$(dirname ${1/upscaled/postprocess})
echo $1
echo $fn
echo ./postprocess/"$fn"

if ! [ -f "$fd"/"$fn" ]; then
 gmic input "$1" iain_nr_2019 2,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.08,0,0,0 afre_sharpenfft 15,1 output "$fd"/"$fn"
fi

python3 ./main.py -i "$fd"/"$fn" -d True -s True -b 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -r 734,1024 -m ./cardmask.png

