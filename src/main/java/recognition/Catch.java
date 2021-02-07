package recognition;

import java.awt.event.KeyEvent;

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

		CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera1.getGamma()); // drawing a window
		Frame capturedFrame = null;   // object to the captured frame

		while ((capturedFrame = camera1.grab()) != null) {
			if(cFrame.isVisible()) {
				cFrame.showImage(capturedFrame);
			}
		}
		cFrame.dispose();  // free memory
		camera1.stop();

	}

}