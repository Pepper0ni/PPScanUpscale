mkdir -p ./temp;
mkdir -p ./upscaled;
mkdir -p ./output;
#for f in ./input/*.png; do
# fn=$(basename "$f")
# echo $fn
# gmic input "$f" iain_nr_2019 1,0,0,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output ./temp/"$fn"
#done

#for f in ./input/*.jpg; do
# fn=$(basename -s .jpg "$f")
# echo $fn
# gmic input "$f" fx_LCE 100,5,0,0.3,5,0 iain_nr_2019 1,0,0,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output ./temp/"$fn".png
#done

python3 /mnt/D/github/Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x2plus -i ./input/ -o ./upscaled/ --fp32 --ext png -s 2 #./debordered/ -o ./upscaled/ --fp32 --ext png -s 2

for f in ./upscaled/*.png; do
 fn=$(basename "$f")
 echo $fn
 gmic input "$f" fx_LCE 100,5,0,0.3,5,0 iain_nr_2019 1,0,0,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output ./temp/"$fn"
done

python3 ./main.py -i ./upscaled/ -o ./output/ -b 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -r 734,1024 -m ./cardmask.png

rm -r ./temp
