echo $1
fb=$(basename "$1")
fd=$(dirname ${1/upscaled\//})
echo $fd
fd=${fd/\/\//\/}
fd=${fd/.\//}
fn=./postprocess/"$fd"/"$fb"
fo=./output/"$fd"/"$fb"
mkdir -p ./output/"$fd";
echo $fd
echo $fn
echo $fo

python3 ./main.py -i $fn -o $fo -b 0.029296875,0.029296875,0.0408719346049,0.0408719346049 -r 734,1024 -e 38,38,38,38 -m ./cardmask.png -cb 2,2,2,2 -f 20,30,134,255,192,255 -ha 3,29,0.0033,3,29,0.0016 -dh 3,19,0.0032,5,29,0.0092 -ct ./cache/"$fd"/ $2
