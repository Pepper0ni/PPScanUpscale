# PPScanUpscale
A series of scripts for upscaling Pokemon Parajis card scans, cleaning them up, and repairing the borders to match the actual cards.

Currently works using bash scripts, but the bulk of the work is done via g'mic and python scripts, so the code should be compatible with windows if the support scripts are ported to it.

## Requirements:

real-ESRGAN

ncnn_vulkan

Joey's fork of ESRGAN (for CUDA)

G'mic (NOT GMIC PY)

scimage

openCV

numpy

## How to Use:

The process can be roughly divided into 3 steps, preperation, upscaling and processing

### Preperation:
Seperate the source images into different border types. They need processing with different settings and thus bash scripts.

Place the Source images in the input folder. 

Make sure that G'Mic is usable on the command line and up to date

Clone real-ESRGAN into an accessable location. This is needed for the upscale step even when using ncnn (as a converted model gave different results, it instead falls back onto CPU mode)

Git for it is here https://github.com/xinntao/Real-ESRGAN

If you are going to be doing the clean and upscale step:

  Download the models for your graphics card type from the links below, and put them in the models folder
  
  Nvidia: https://drive.google.com/file/d/1Ks-9T8SIxQWIK5SG9E0hizEfBVvW26-l/view?usp=sharing
  
  AMD/Intel: https://drive.google.com/file/d/1G2N5BmgI6IVrH-i7SQRDf1qWXkO30-Ox/view?usp=sharing
  
  If you are using a Nvidia GPU, clone joey's fork of ESRGANinto an accessible location.

  Find it here: https://github.com/joeyballentine/ESRGAN

### Upscaling:
Upscaling is a multi-step process where the scan is denoised, dedithered and then 2x upscaled. it does this using a custom script that, depending on the method chosen, either handles the first 2 steps itself or hooks into an old ESRGAN fork, and then hooks into real-ESRGAN to for the upscale proper.

Run either UpscaleOnlyCUDA.bash or UpscaleOnlyNcnn.bash with bash for Nvidia and other GPUs respectivly to just upscale things.

The upscaled results will be placed in ./upscaled

running Process<border type>.bash script will run this script and the next one consecutivly

### Processing:
To turn an upscaled scan into a card, another multi-step process is used. The exact process varies based on the border/generation of the card, but for now the base-neo process will be described.

First the images are sorted based on the 1% low value of thier text, in order to try and avoid making already dark images darker. darkness first edition cards are also seperated to avoid issues with the first edition logo bleeding into blobs. 

The cards are then run through iain's denoiser and a darkness only local contrast enhancment, with different settings for each seperation of card, to further denoise athe image and darken the blacks slightly. 

After that the main process uses edge detection to find the inner border, and shapes that to the correct size and shape before blurring the border while sharpening the non-border

This is handled within the border appropriote "PostUpscale" bash scripts and thier components.


