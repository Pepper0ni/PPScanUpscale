mkdir -p ./resized;
mkdir -p ./output;
mkdir -p ./deskewed;

python3 ./main.py -i ./upscaled/ -d ./deskewed/ -b 12,12,34,12 -e 0.0390625,0.0390625,0.0456403269755,0.0456403269755 #-ef 42,42,42,42 #-ef 50,50,50,50

for f in ./deskewed/*.png; do
 fn=$(basename "$f")
 echo $fn
 gmic input "$f" solidify 4,1,5 resize 734,1024,100%,100%,5 output ./resized/"$fn"
done

python3 maskcorners.py
