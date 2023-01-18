mkdir -p ./denoise;
mkdir -p ./dedither;
mkdir -p ./upscaled;

python3 "$2"upscale.py ./models/1x_ISO_denoise_v1.pth -i ./input/ -o ./denoise/ -se

python3 "$2"upscale.py ./models/1x_artifacts_dithering_alsa.pth -i ./denoise/ -o ./dedither/ -se

rm -r ./denoise

python3 upscaleimage.py -i ./dedither/ -o ./upscaled/ -m ./models/ -e "$1"

rm -r ./dedither
