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

python3 ./main.py -i $fn -o $fo -b 0.0297,0.0297,0.0414,0.0414 -r 734,1024 -e 38,38,38,38 -m ./cardmask.png -f 18,139,20,255,97,255 -tf 0,177,0,67,202,255 -dh 3,29,0.0028,3,29,0.0028 -rb True -ex True -fc 175 -cb 2,2,2,2 -bb 1,1 -ct ./cache/"$fd"/ -br ./exBorders $2

# 15,139,12,255,97,255
