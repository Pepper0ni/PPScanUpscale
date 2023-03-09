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
  fc=./clean/"$fb"
 elif [[ $fd == "" ]]; then
  fn=./postprocess/"$fb"
  fc=./clean/"$fb"
 else
  fn=./postprocess/"$fd"/"$fb"
  fc=./clean/"$fd"/"$fb"
 fi
 echo "$fd"
 echo "$fn"
 echo "$fc"
 mkdir -p ./postprocess/"$fd";
 mkdir -p ./clean/"$fd";
 if ! [ -f "$fn" ]; then
  gmic input "$f" iain_nr_2019 2,0,0,0,0.5,1,0,5,2,0,2,0,1,0,0 output "$fn"
 fi
done
for p in $(find ./postprocess/ -name "*.png" -type f); do
 echo $p
 pb=$(basename "$p")
 pc=$(dirname ${p/postprocess/clean})
 if ! [ -f "$pc"/"$pb" ]; then
  gmic input "$p" iain_nr_2019 2,0,0,0,0.5,1,0,5,2,0,2,0,1,0,0 output "$pc"/"$pb"
 fi
done
