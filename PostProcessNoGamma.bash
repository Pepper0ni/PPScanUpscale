echo $1
fb=$(basename "$1")
#fp=$(dirname ${1/upscaled/temp})
fd=$(dirname ${1/upscaled\//})
fd=${fd/dark/}
fd=${fd/\/\//\/}
fd=${fd/.\//}
if [[ $fd == "." ]]; then
 fn=./postprocess/"$fb"
elif [[ $fd == "" ]]; then
 fn=./postprocess/"$fb"
else
 fn=./postprocess/"$fd"/"$fb"
fi
echo "$fd"
echo "$fn"
mkdir -p ./postprocess/"$fd";
if ! [ -f "$fn" ]; then
 gmic input "$1" iain_nr_2019 0.05,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.04,0,0,0 "$fn"
fi

