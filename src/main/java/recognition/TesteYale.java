package recognition;

import java.io.File;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class TesteYale {
    public static void main(String[] args) {
        int totalHits = 0;
        double percentageHits = 0;
        double totalConfidence = 0;
        
       // FaceRecognizer recognizer = createEigenFaceRecognizer();
        //FaceRecognizer recognizer = createFisherFaceRecognizer();
        FaceRecognizer recognizer = createLBPHFaceRecognizer();

       // recognizer.load("src\\main\\java\\resources\\classifierEigenFacesYale.yml");
        //recognizer.load("src\\main\\java\\resources\\classifierFisherFacesYale.yml");
        recognizer.load("src\\main\\java\\resources\\classifierLBPHYale.yml");

        File directory = new File("src\\main\\java\\yalefaces\\test");
        File[] files = directory.listFiles();
        
        for (File image : files) {           
            Mat photo = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int personId = Integer.parseInt(image.getName().substring(7, 9));          
            resize(photo, photo, new opencv_core.Size(160, 160));

            IntPointer label = new IntPointer(1);
            DoublePointer confidence = new DoublePointer(1);
            recognizer.predict(photo, label, confidence);
            int selection = label.get(0);
            System.out.println(personId + " was recognized as " + selection + " - " + confidence.get(0));
            if (personId == selection) {
                totalHits++;
                totalConfidence += confidence.get(0);
            }
        }
        
        percentageHits = (totalHits / 30.0) * 100;
        totalConfidence = totalConfidence / totalHits;
        System.out.println("Total hits: " + totalHits + " faces from  30 faces");
        System.out.println("Percentage of correct answers: " + percentageHits);
        System.out.println("Total confidence: " + totalConfidence);
    }
}