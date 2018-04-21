import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.File

//Path to the source folder
val src = "/mnt/host/c/Users/Sayed/bdh/images/"
//Path to the destination folder
val dst = "/mnt/host/c/Users/Sayed/bdh/images_masked/"

//Path to the mask image
val mask = "/mnt/host/c/Users/Sayed/bdh/mask.png"

val w = 1024
val h = 1024

val files = new File(src).listFiles
val overlay = ImageIO.read(new File(mask));


for (file <- files) { 
    val in = ImageIO.read(file)
    val combined = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
    val g = combined.getGraphics();
    g.drawImage(in, 0, 0, null);
    g.drawImage(overlay, 0, 0, null);
    
    ImageIO.write(combined, "png", new File(dst+file.getName()))
    
}
