echo $1
fb=$(basename "$1")
#fp=$(dirname ${1/upscaled/temp})
fd=$(dirname ${1/upscaled\//})
fd=$(dirname ${fd/light\//})
fd=${fd/.\//}
if [[ $fd == "." ]]; then
 fn=./light/"$fb"
else
 fn=./light/"$fd"/"$fb"
fi
echo "$fn"
mkdir -p "$fd";
if ! [ -f "$fn" ]; then
 gmic input "$1" iain_nr_2019 2,1,-0.5,0.5,0.6,1,0,3,0,0,2,0,1,0,0 fx_LCE 200,2,0.08,0,0,0 afre_sharpenfft 15,1 output "$fn"
 #gmic input "$f" fx_LCE 30,4,1,1,32,0 fx_smooth_bilateral 1,7.5,5,0,0 iain_nr_2019 1,0.5,0.5,0,0.5,1,0,25,30,30,7,0,0.5,0,0 output "$fp"/"$fn"
fi

