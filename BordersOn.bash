mkdir -p ./output;

python3 ./main.py -i ./upscaled/ -o ./output/ -b 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -r 734,1024 -m ./cardmask.png #-ef 42,42,42,42 #-ef 50,50,50,50

