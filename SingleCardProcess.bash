fn=$(basename "$1")
echo $fn
#gmic input "$1" fx_LCE 100,5,0,0.3,5,0 iain_nr_2019 1,0,0,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output ./temp/"$fn"

python3 ./main.py -i "$1" -c ./temp/"$fn" -o ./outputc/"$fn" -b 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -r 734,1024 -m ./cardmask.png #-e 42,42,42,42


