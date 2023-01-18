echo $1
fb=$(basename "$1")
fn=./postprocess/"$fb"
fo=./output/"$fb"
echo $fn
echo $fo

python3 ./main.py -i $fn -o $fo -b 0.0390625,0.0390625,0.0456403269755,0.0456403269755 -r 734,1024 -m ./cardmask.png -e 60,54,54,54 $2
