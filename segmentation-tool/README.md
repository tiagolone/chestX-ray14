# Chest X- Ray Segmentation tool
The chest x-ray segmentation script masks the lungs area of the images in the Chest X-Ray data set

# Prerequisites 
 Scala version 2.10.7

# Steps:

- copy mask_images.scala and mask.png to the server
- open mask_images.scala to edit 
    - set the path of the x-ray image directory
    - set the path of the masked images directory
    - set the path of the mask image file (mask.png)
- open sbt console using  the command "sbt console"
- Run mask_images.scala using the command ":load mask_images.scala"








