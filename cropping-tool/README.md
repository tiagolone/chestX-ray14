# Chest X- Ray cropping tool
The chest x-ray cropping script reuces the size of the x-ray images from 1024x1024 to 768x768 by cropping the edges of the images

# Prerequisites 
 Scala version 2.10.7

# Steps:

- copy crop_images.scala to the server
- create a destination directory for the cropped images e.g. cropped_images
- open crop_images.scala to edit 
    - set the path of the x-ray images directory
    - set the path of the destination directory  for the cropped images
- open sbt console using the command "sbt console"
- run crop_images.scala using the command ":load crop_images.scala"








