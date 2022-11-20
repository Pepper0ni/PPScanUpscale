fn=$(basename "$1")
echo $fn

if [./postprocess/"$fn"]; then
 if [./temp/"$fn"]; then
  gmic input "$f" iain_nr_2019 2,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.08,0,0,0 afre_sharpenfft 15,1 output ./postprocess/"$fn" fx_LCE 100,5,0,0.3,5,0 fx_LCE 100,5,0,0.3,16,0 iain_nr_2019 1,0,0,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output ./temp/"$fn"
 fi
fi

python3 ./main.py -i ./postprocess/"$fn" -o ./output/"$fn" -c ./temp/"$fn" -b 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -r 734,1024 -m ./cardmask.png
