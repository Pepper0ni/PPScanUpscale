# PPScanUpscale
A series of scripts for upscaling Pokemon Parajis card scans, cleaning them up, and repairing the borders to match the actual cards.

Currently works using bash scripts, but the bulk of the work is done via g'mic and python scripts, so the code should be compatible with windows if the support scripts are ported to it.

Reuirements:
real-ESRGAN
ncnn_vulkan
Joey's fork of ESRGAN (for CUDA)
G'mic (NOT GMIC PY)
scimage
openCV
numpy

How to Use:

The process can be roughly divided into 3 steps, preperation, upscaling and processing

Preperation:
Seperate the source images into different border types. They need processing with different settings and thus bash scripts.
Place the Source images in the input folder. 
Clone real-ESRGAN into an accessable location. this is needed for the upscale setp even when using ncnn (as a converted model gave different results, it instead falls back onto CPU mode)

If you are going to be doing the clean and upscale step:
  Download the models for your graphics card type from the links below, and put them in the models folder
  Nvidia: https://drive.google.com/file/d/1Ks-9T8SIxQWIK5SG9E0hizEfBVvW26-l/view?usp=sharing
  AMD/Intel: https://drive.google.com/file/d/1G2N5BmgI6IVrH-i7SQRDf1qWXkO30-Ox/view?usp=sharing
  


Upscaling:
Upscaling is a multi-step process where the 
