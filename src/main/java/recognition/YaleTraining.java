package recognition;

import java.io.File;
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class YaleTraining {
    public static void main(String[] args) {
        File directory = new File("src\\main\\java\\yalefaces\\training");
        File[] files = directory.listFiles();
        MatVector photos = new MatVector(files.length);
        Mat labels = new Mat(files.length, 1, CV_32SC1);
        IntBuffer bufferLabels = labels.createBuffer();
        int counter = 0;
      
        for (File image : files) {
            Mat photo = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int personId = Integer.parseInt(image.getName().substring(7,9));
            resize(photo, photo, new Size(160, 160));
            photos.put(counter, photo);
            bufferLabels.put(counter, personId);
            counter++;
        }
        
        //classifiers
        FaceRecognizer eigenface = createEigenFaceRecognizer(30, 0);
        FaceRecognizer fisherface = createFisherFaceRecognizer(30, 0);        
        FaceRecognizer lbph = createLBPHFaceRecognizer(12, 10, 15, 15, 0);

        
        // classifiers -> learning-training to generate yml codes
     	eigenface.train(photos, labels);
     	eigenface.save("src\\main\\java\\resources\\classifierEigenFacesYale.yml");
     		
     	fisherface.train(photos, labels);
     	fisherface.save("src\\main\\java\\resources\\classifierFisherFacesYale.yml");
     	
     	lbph.train(photos, labels);
     	lbph.save("src\\main\\java\\resources\\classifierLBPHYale.yml");
    }
}