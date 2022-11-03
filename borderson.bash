for f in ./deskewed/*.png; do
 fn=$(basename "$f")
 echo $fn
 gmic input "$f" solidify 5,2,5 resize 734,1024,100%,100%,5 output ./resized/"$fn"
done

python3 maskcorners.py
