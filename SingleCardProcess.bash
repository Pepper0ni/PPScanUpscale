#mkdir -p ./temp;
mkdir -p ./output;
mkdir -p ./postprocess;

fn=$(basename "$1")
echo $fn

if [./postprocess/"$fn"]; then
 #if [./temp/"$fn"]; then
  gmic input "$f" iain_nr_2019 2,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.08,0,0,0 afre_sharpenfft 15,1 output ./postprocess/"$fn"
  #gmic input "$f" fx_LCE 30,4,1,1,32,0 fx_smooth_bilateral 1,7.5,5,0,0 iain_nr_2019 1,0.5,0.5,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output ./temp/"$fn"
 #fi
fi

python3 ./main.py -i ./postprocess/"$fn" -o ./output/"$fn" -b 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -r 734,1024 -m ./cardmask.png #-c ./temp/"$fn"
