mkdir -p ./dedither;
mkdir -p ./upscaled;

python3 upscaleimagencnn.py -i ./input/ -o ./dedither/ -m ./models/

python3 upscaleimage.py -i ./dedither/ -o ./upscaled/ -m ./models/2x_Pkmncards_PP_Dubu_RealESRGAN.pth -e "$1"

#rm -r ./dedither
