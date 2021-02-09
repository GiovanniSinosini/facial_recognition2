package recognition;

import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;

public class Training {

	public static void main(String[] args) {
		File directory = new File ("src\\main\\java\\photos");
		FilenameFilter imageFilter = new FilenameFilter() {   // filter image type
			public boolean accept(File dir, String name) {
				return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png") ;
			}
		}; 

		File[] files = directory.listFiles(imageFilter);    // vector to store the images according to the filter
		MatVector photos = new MatVector(files.length);  // save archived photos
		Mat labels = new Mat(files.length, 1 , opencv_core.CV_32SC1);  //record name/labels of photos
		IntBuffer bufferLabels = labels.createBuffer();  //to store the labels
		int counter = 0;  //count images

		for( File image : files) {   // fill in the data to train classifiers
			Mat photo = opencv_imgcodecs.imread(image.getAbsolutePath(), opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);   //take the image by name and convert to gray scale
			int personId = Integer.parseInt(image.getName().split("\\.")[1]);  // search person id

			opencv_imgproc.resize(photo, photo, new Size(160,160));
			photos.put(counter, photo);   //search photo
			bufferLabels.put(counter, personId);  //search person id
			counter++;
		}

		//classifiers
		FaceRecognizer eigenFaces = createEigenFaceRecognizer();
		FaceRecognizer fisherFaces = createFisherFaceRecognizer();
		FaceRecognizer lbph = createLBPHFaceRecognizer();

		// classifiers -> learning-training to generate yml codes
		eigenFaces.train(photos, labels);
		eigenFaces.save("src\\main\\java\\resources\\classifierEigenFaces.yml");
		fisherFaces.train(photos, labels);
		fisherFaces.save("src\\main\\java\\resources\\classifierFisherFaces.yml");
		lbph.train(photos, labels);
		lbph.save("src\\main\\java\\resources\\classifierLBPH.yml");
	}
}