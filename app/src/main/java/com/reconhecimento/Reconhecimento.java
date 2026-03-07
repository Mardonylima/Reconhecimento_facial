package com.reconhecimento;

import java.awt.event.KeyEvent;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_PLAIN;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class Reconhecimento {
    
    //@SuppressWarnings("null")

    private static final java.util.logging.Logger logger = java.util.logging.Logger.getLogger(Reconhecimento.class.getName());

    public static void main(String[] args) throws FrameGrabber.Exception, InterruptedException{
        
        // Inicializa a captura de vídeo
        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        String[] pessoas = {"", "Mardony", "Mardenya"};
        camera.start();

        // Carregar o classificador pré-treinado para detecção facial
        CascadeClassifier detectorFace = new CascadeClassifier("src/main/java/com/resource/haarcascade_frontalface_alt.xml");
        FaceRecognizer reconhecedor = EigenFaceRecognizer.create();
        reconhecedor.read("src/main/java/com/resource/treinamentoEigenfaces.yml");
        // Criar a janela para exibir o vídeo
        CanvasFrame cFrame = new CanvasFrame("Reconhecimento Facial", CanvasFrame.getDefaultGamma() / camera.getGamma());
        Frame frameCapturado = null;
        Mat imagemColorida = null;
        
            // Loop para capturar e processar os frames da câmera
            while ((frameCapturado = camera.grab()) != null) {
                imagemColorida = converteMat.convert(frameCapturado);
                Mat imagemCinza = new Mat();
                cvtColor(imagemColorida, imagemCinza, COLOR_BGR2GRAY);
                // Detectar faces na imagem em escala de cinza
                RectVector facesDetectadas = new RectVector();
                detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
                // Desenhar retângulos ao redor das faces detectadas
                for (int i = 0; i < facesDetectadas.size(); i++) {
                    Rect dadosFace = facesDetectadas.get(i);
                    rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0)); 
                    // Extrair a face capturada
                    Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                    resize(faceCapturada, faceCapturada, new Size(160, 160));
                    // Realizar o reconhecimento facial
                    IntPointer rotulo = new IntPointer(1);
                    DoublePointer confiança = new DoublePointer(1);
                    reconhecedor.predict(faceCapturada, rotulo, confiança);
                    int predicao = rotulo.get(0);
                    String nome; 
                    if (predicao == -1) {
                        nome = "Desconhecido";
                    } else {
                        nome = pessoas[predicao] + " - " + String.format("%.2f", confiança.get(0));
                    }
                    // Exibir o nome da pessoa reconhecida na imagem
                    int X = Math.max(dadosFace.tl().x() - 10, 0);
                    int Y = Math.max(dadosFace.tl().y() - 10, 0);
                    putText(imagemColorida, nome, new Point(X, Y), FONT_HERSHEY_PLAIN, 1.5, new Scalar(0, 255, 0, 0), 2, 8, false);
                }
                if (cFrame.isVisible()) {
                    cFrame.showImage(frameCapturado);
                }
            }

        // Liberar recursos
        camera.stop();
        cFrame.dispose();
    }
}
