package com.reconhecimento;

import java.awt.event.KeyEvent;
import java.util.Scanner;
import java.util.logging.Level;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class Capture {
    
    //@SuppressWarnings("null")

    private static final java.util.logging.Logger logger = java.util.logging.Logger.getLogger(Capture.class.getName());

    public static void main(String[] args) throws FrameGrabber.Exception, InterruptedException{
        
        // Inicializa a captura de vídeo
        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        camera.start();

        // Carregar o classificador pré-treinado para detecção facial
        CascadeClassifier detectorFace = new CascadeClassifier("src/main/java/com/resource/haarcascade_frontalface_alt.xml");

        // Criar a janela para exibir o vídeo
        CanvasFrame cFrame = new CanvasFrame("Reconhecimento Facial", CanvasFrame.getDefaultGamma() / camera.getGamma());
        Frame frameCapturado = null;
        Mat imagemColorida = null;
        Mat imagemCinza = new Mat();
        int numeroAmostras = 25;
        int amostra = 1;
        // Solicitar o ID do usuário para nomear as imagens capturadas
        logger.log(Level.INFO,"Solicitando ID do usuário: ");
        Scanner cadastro = new Scanner(System.in); 
        int idPessoa = cadastro.nextInt();
            // Loop para capturar e processar os frames da câmera
            while ((frameCapturado = camera.grab()) != null) {
                imagemColorida = converteMat.convert(frameCapturado);
                cvtColor(imagemColorida, imagemCinza, COLOR_BGR2GRAY);
                // Detectar faces na imagem em escala de cinza
                RectVector facesDetectadas = new RectVector();
                detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
                if (tecla == null){
                    tecla = cFrame.waitKey(5);
                }
                // Desenhar retângulos ao redor das faces detectadas
                for (int i = 0; i < facesDetectadas.size(); i++) {
                    Rect dadosFace = facesDetectadas.get(i);
                    rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0)); 
                    // Extrair a face capturada
                    Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                    resize(faceCapturada, faceCapturada, new Size(160, 160));
                    if (tecla == null){
                        tecla = cFrame.waitKey(5);
                    }
                    // Verificar se a tecla 'q' foi pressionada para capturar a foto
                    if (tecla != null && tecla.getKeyChar() == 'q' && amostra <= numeroAmostras) {
                        // Salvar a face capturada como uma imagem
                        imwrite("src/main/java/com/fotos/pessoa." + idPessoa + "." + amostra + ".jpg", faceCapturada);
                        logger.log(Level.INFO,"Foto {0} capturada com sucesso\n", amostra);
                        amostra++;
                            
                    }
                    tecla = null;
                       
                }
                if (tecla == null){
                    tecla = cFrame.waitKey(20);
                }
                if (cFrame.isVisible()) {
                    cFrame.showImage(frameCapturado);
                }

                if (amostra > numeroAmostras) {
                    break;
                }
            }

        // Liberar recursos
        camera.stop();
        cFrame.dispose();
    }
}
