package com.reconhecimento;

import java.awt.event.KeyEvent;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class Capture {
    public static void main(String[] args) throws FrameGrabber.Exception{
        
        // Inicializa a captura de vídeo
        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        FrameGrabber camera = FrameGrabber.createDefault(0);
        camera.start();

        // Carregar o classificador pré-treinado para detecção facial
        CascadeClassifier detectorFace = new CascadeClassifier("src/main/java/com/resource/haarcascade_frontalface_alt.xml");

        // Criar a janela para exibir o vídeo
        CanvasFrame cFrame = new CanvasFrame("Reconhecimento Facial", CanvasFrame.getDefaultGamma() / camera.getGamma());
        Frame frameCapturado = null;
        Mat imagemColorida = new Mat();
            // Loop para capturar e processar os frames da câmera
            while ((frameCapturado = camera.grab()) != null) {
                imagemColorida = converteMat.convert(frameCapturado);
                Mat imagemCinza = new Mat();
                cvtColor(imagemColorida, imagemCinza, COLOR_BGR2GRAY);
                RectVector facesDetectadas = new RectVector();
                detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
                // Desenhar retângulos ao redor das faces detectadas
                for (int i = 0; i < facesDetectadas.size(); i++) {
                    Rect dadosFace = facesDetectadas.get(i);
                    rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0));    
                }
                if (cFrame.isVisible()) {
                    cFrame.showImage(frameCapturado);
                }
                 else {
                    break;
                }
            }

        // Liberar recursos
        camera.stop();
        cFrame.dispose();
    }
}
