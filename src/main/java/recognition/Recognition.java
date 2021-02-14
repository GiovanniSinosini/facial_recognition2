package recognition;

import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

public class Recognition {
	public static void main(String[] args) throws Exception, InterruptedException {
		OpenCVFrameConverter.ToMat convertMat = new OpenCVFrameConverter.ToMat(); // convert image to Mat
		OpenCVFrameGrabber camera1 = new OpenCVFrameGrabber(0); // capturing webcam images
		String[] people = {"", "Giovanni", "Lia", "Priscila"};
		camera1.start();

		CascadeClassifier faceDetector = new CascadeClassifier("src\\main\\java\\resources\\haarcascade_frontalface_alt.xml");
		
		//FaceRecognizer recognizer = createEigenFaceRecognizer();  // classifier
		//recognizer.load("src\\main\\java\\resources\\classifierEigenFaces.yml");
		// recognizer.setThreshold(8000);      // trust number  
		
		//FaceRecognizer recognizer = createFisherFaceRecognizer();
		//recognizer.load("src\\main\\java\\resources\\classifierFisherFaces.yml");
		
		FaceRecognizer recognizer = createLBPHFaceRecognizer();
		recognizer.load("src\\main\\java\\resources\\classifierLBPH.yml");
		recognizer.setThreshold(65.0);
		
		CanvasFrame cFrame = new CanvasFrame("Recognition", CanvasFrame.getDefaultGamma() / camera1.getGamma()); // drawing a window
		Frame capturedFrame = null; // object to the captured frame
		Mat colorImage = new Mat(); // transfer from frame to color image for face detection
		int sampleNumber = 30;
		int sample = 1;

		while ((capturedFrame = camera1.grab()) != null) {
			colorImage = convertMat.convert(capturedFrame);
			Mat grayImage = new Mat();
			opencv_imgproc.cvtColor(colorImage, grayImage, opencv_imgproc.COLOR_BGRA2GRAY); // convert image to gray for better detection
			RectVector detectedFaces = new RectVector(); // store detected faces
			faceDetector.detectMultiScale(grayImage, detectedFaces, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));

			for (int i = 0; i < detectedFaces.size(); i++) { // cycle detected faces vector
				Rect faceData = detectedFaces.get(i);
				opencv_imgproc.rectangle(colorImage, faceData, new Scalar(0, 0, 255, 0)); // insert rectangle in color
																							// image
				Mat capturedface = new Mat(grayImage, faceData);
				opencv_imgproc.resize(capturedface, capturedface, new Size(160, 160));
				
				IntPointer label = new IntPointer(1); // identify the image label
				DoublePointer confidence = new DoublePointer(1); 
				recognizer.predict(capturedface, label, confidence); // will classify the new image according to the training
				int selection = label.get(0); // choice made by the classifier
				String name;
				if (selection == -1) {
					name="Unknown";
				} else {
					name= people[selection] + " - " + confidence.get(0);
				}
				
				// entering the name on the screen
				int x = Math.max(faceData.tl().x() -10, 0); 
				int y = Math.max(faceData.tl().y() -10, 0);
				opencv_imgproc.putText(colorImage, name, new Point(x,y), opencv_core.FONT_HERSHEY_PLAIN, 1.4, new Scalar(0,255,0,0));

			}

			if (cFrame.isVisible()) {
				cFrame.showImage(capturedFrame);
			}

			if (sample > sampleNumber) {
				break;
			}
		}
		cFrame.dispose(); // free memory
		camera1.stop();
		camera1.close();
		colorImage.close();
		faceDetector.close();
	}
}