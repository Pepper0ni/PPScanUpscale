python3 getBorders.py

for p in $(find ./exBorders/top/ -name "*.png" -type f); do
 gmic input "$p" solidify 0,2,0 iain_nr_2019 2,0,0,0,0.5,1,0,10,5,0,7,0,1,0,0 output "$p"
done

for p in $(find ./exBorders/bottom/ -name "*.png" -type f); do
 gmic input "$p" solidify 0,2,0 iain_nr_2019 2,0,0,0,0.5,1,0,10,5,0,7,0,1,0,0 output "$p"
done

for p in $(find ./exBorders/left/ -name "*.png" -type f); do
 gmic input "$p" solidify 0,2,0 iain_nr_2019 2,0,0,0,0.5,1,0,10,5,0,7,0,1,0,0 output "$p"
done

for p in $(find ./exBorders/right/ -name "*.png" -type f); do
 gmic input "$p" solidify 0,2,0 iain_nr_2019 2,0,0,0,0.5,1,0,10,5,0,7,0,1,0,0 output "$p"
done
