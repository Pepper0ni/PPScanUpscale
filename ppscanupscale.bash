mkdir -p ./debordered;
mkdir -p ./temp;
mkdir -p ./upscaled;
mkdir -p ./resized;
mkdir -p ./output;
#for f in ./input/*.png; do
# fn=$(basename "$f")
# echo $fn
# gmic input "$f" iain_nr_2019 1,0,0,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output ./temp/"$fn"
#done
for f in ./input/*.jpg; do
 fn=$(basename -s .jpg "$f")
 echo $fn
 gmic input "$f" fx_LCE 100,5,0,0.3,5,0 iain_nr_2019 1,0,0,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output ./temp/"$fn".png
done
python3 ./main.py
python3 /mnt/D/github/Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x2plus -i ./debordered -o ./upscaled --fp32 --ext png -s 2
for f in ./upscaled/*.png; do
 fn=$(basename "$f")
 echo $fn
 gmic input "$f" resize 667,961,100%,100%,5 output ./resized/"$fn"
done
