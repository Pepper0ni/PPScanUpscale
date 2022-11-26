echo $1
fb=$(basename "$1")
#fp=$(dirname ${1/upscaled/temp})
fd=$(dirname ${1/upscaled\//})
fd=$(dirname ${fd/dark\//})
fd=${fd/.\//}
if [[ $fd == "." ]]; then
 fn=./dark/"$fb"
else
 fn=./dark/"$fd"/"$fb"
fi
echo "$fn"
mkdir -p "$fd";
if ! [ -f "$fn" ]; then
 gmic input "$1" iain_nr_2019 0.05,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.04,0,0,0 afre_sharpenfft 15,1 output "$fn"
fi

