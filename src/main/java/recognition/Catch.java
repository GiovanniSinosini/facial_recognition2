package recognition;

import java.awt.event.KeyEvent;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

public class Catch {
	
	public static void main(String[] args) throws Exception {
		KeyEvent keyboradKey = null;   // catch keyborad key
		OpenCVFrameConverter.ToMat convertMat = new OpenCVFrameConverter.ToMat();    // convert image to Mat
		OpenCVFrameGrabber camera1 = new OpenCVFrameGrabber(0);						// capturing webcam images

		camera1.start();

		CascadeClassifier faceDetector = new CascadeClassifier("src\\main\\java\\resources\\haarcascade_frontalface_alt.xml");
		CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera1.getGamma()); // drawing a window
		Frame capturedFrame = null;   // object to the captured frame
		Mat colorImage = new Mat();   // transfer from frame to color image for face detection

		while ((capturedFrame = camera1.grab()) != null) {
			colorImage = convertMat.convert(capturedFrame);
			Mat grayImage = new Mat();               
			opencv_imgproc.cvtColor(colorImage, grayImage, 1); //convert image to gray for better detection
			RectVector detectedFaces = new RectVector();      //store detected faces
			faceDetector.detectMultiScale(grayImage, detectedFaces, 1.1, 1, 0, new Size(150,150), new Size(500,500));
			for (int i = 0; i < detectedFaces.size(); i++) {   // cycle detected faces vector
				Rect faceData = detectedFaces.get(0);
				opencv_imgproc.rectangle(colorImage, faceData, new Scalar(0,0,255, 0));  // insert rectangle in color image
			}

			if(cFrame.isVisible()) {
				cFrame.showImage(capturedFrame);
			}
		}
		cFrame.dispose();  // free memory
		camera1.stop();
		
	}
}