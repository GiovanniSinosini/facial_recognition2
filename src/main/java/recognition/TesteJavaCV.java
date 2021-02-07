package recognition;

import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;

public class TesteJavaCV {

	public static void main(String args[]) {

		FaceRecognizer r = createEigenFaceRecognizer();
		System.out.print("Hello AI");
		
	}
}
