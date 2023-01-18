echo $1
for f in $(find $1 -name "*.png" -type f); do
 echo $f
 fb=$(basename "$f")
 fd=$(dirname ${f/upscaled\//})
 fd=${fd/light/}
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
  gmic input "$f" iain_nr_2019 2,0,0,0,0.5,1,0,5,2,0,2,0,1,0,0 output "$fn"
 fi
done
