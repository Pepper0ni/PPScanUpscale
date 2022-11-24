#mkdir -p ./temp;
mkdir -p ./output;
mkdir -p ./postprocess;
mkdir -p ./light/;
mkdir -p ./dark/;

python3 ./splitByValue.py -i ./upscaled

for f in $(find ./light -name "*.png" -type f); do
 echo $f
 fn=$(basename "$f")
 #fp=$(dirname ${f/upscaled/temp})
 fd=$(dirname ${f/light/postprocess})
 #mkdir -p "$fp";
 mkdir -p "$fd";
 if ! [ -f "$fd"/"$fn" ]; then
  gmic input "$f" iain_nr_2019 2,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.08,0,0,0 afre_sharpenfft 15,1 output "$fd"/"$fn"
  #gmic input "$f" fx_LCE 30,4,1,1,32,0 fx_smooth_bilateral 1,7.5,5,0,0 iain_nr_2019 1,0.5,0.5,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output "$fp"/"$fn"
 fi
done

for f in $(find ./dark -name "*.png" -type f); do
 echo $f
 fn=$(basename "$f")
 #fp=$(dirname ${f/upscaled/temp})
 fd=$(dirname ${f/dark/postprocess})
 #mkdir -p "$fp";
 mkdir -p "$fd";
 if ! [ -f "$fd"/"$fn" ]; then
  gmic input "$f" iain_nr_2019 0.5,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.08,0,0,0 afre_sharpenfft 15,1 output "$fd"/"$fn"
  #gmic input "$f" fx_LCE 30,4,1,1,32,0 fx_smooth_bilateral 1,7.5,5,0,0 iain_nr_2019 1,0.5,0.5,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output "$fp"/"$fn"
 fi
done


python3 ./main.py -i ./postprocess/ -o ./output/ -b 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -r 734,1024 -m ./cardmask.png -ma True

#rm -r ./temp
rm -r ./light
rm -r ./dark
