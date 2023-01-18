mkdir -p ./upscaled;

python3 upscaleimage.py -i ./input/ -o ./upscaled/ -m ./models/ -e "$1" -n True

