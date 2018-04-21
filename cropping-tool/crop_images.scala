import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.File

// source director
val src = "/mnt/host/c/Users/Sayed/bdh/images/"

//destination directory
val dst = "/mnt/host/c/Users/Sayed/bdh/images_cropped/"

val x = 128
val y = 128
val w = 768
val h = 768

val files = new File(src).listFiles


for (file <- files) { 
    val in = ImageIO.read(file)
    val out = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    out.getGraphics().drawImage(in, 0, 0, w, h, x, y, x + w, y + h, null)
    ImageIO.write(out, "png", new File(dst+file.getName()))
    
}