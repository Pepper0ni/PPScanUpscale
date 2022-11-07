fn=$(basename "$1")
echo $fn
gmic input "$1" fx_LCE 100,5,0,0.3,5,0 iain_nr_2019 1,0,0,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output ./temp/"$fn"

python3 ./singlecard.py -i "$1" -c ./temp/"$fn" -d ./deskewed/"$fn" -b 12,12,34,12 -e 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -ef 42,42,42,42 -de True

gmic input ./deskewed/"$fn" solidify 4,1,5 resize 734,1024,100%,100%,5 output ./resized/"$fn"

python3 maskcorners.py -i ./resized/"$fn" -o ./output/"$fn" -s True

