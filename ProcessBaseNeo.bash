mkdir -p ./temp;
mkdir -p ./upscaled;
mkdir -p ./output;
mkdir -p ./denoise;
mkdir -p ./dedither;

python3 "$2"upscale.py ./models/1x_ISO_denoise_v1.pth -c -i ./input/ -o ./denoise/

python3 "$2"upscale.py ./models/1x_artifacts_dithering_alsa.pth -c -i ./denoise/ -o ./dedither/

python3 upscaleimage.py -i ./dedither/ -o ./upscaled/ -m ./models/2x_Pkmncards_PP_Dubu_RealESRGAN.pth -e "$1"

for f in $(find ./upscaled -name "*.png" -type f); do
 echo $f
 fn=$(basename "$f")
 fp=$(dirname ${f/upscaled/temp})
 echo $fn
 echo $fp
 mkdir -p "$fp";
 gmic input "$f" fx_LCE 100,5,0,0.3,5,0 iain_nr_2019 1,0,0,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output "$fp"/"$fn"
done

python3 ./main.py -i ./upscaled/ -o ./output/ -b 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -r 734,1024 -m ./cardmask.png

rm -r ./temp
