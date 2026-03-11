package com.reconhecimento;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import java.util.logging.Level;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;

public class Treinamento {
    
    private static final java.util.logging.Logger logger = java.util.logging.Logger.getLogger(Treinamento.class.getName());
    
    public static void main(String[] args) {
        File diretorio = new File(System.getProperty("user.dir") + "/src/main/java/com/fotos");
        FilenameFilter filtroImagem = (dir, nome) -> nome.endsWith(".jpg") || nome.endsWith(".gif") || nome.endsWith(".png");
        File[] arquivos = diretorio.listFiles(filtroImagem);
        MatVector fotos = new MatVector(arquivos.length);
        Mat rotulos = new Mat(arquivos.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;   
        for (File imagem: arquivos) {
            Mat foto = imread(imagem.getAbsolutePath(), IMREAD_GRAYSCALE);
                if (foto.empty()) {
                    logger.log(Level.SEVERE, () -> "Erro ao carregar a imagem: " + imagem.getAbsolutePath());
                    continue;
                }
                else {
                    logger.log(Level.INFO, () -> "Imagem carregada com sucesso: " + imagem.getAbsolutePath());
                }
            int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);
            resize(foto, foto, new Size(160,160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);
            contador++;            
        }

        FaceRecognizer eigenfaces = EigenFaceRecognizer.create();
        FaceRecognizer fisherfaces = FisherFaceRecognizer.create();
        FaceRecognizer lbph = LBPHFaceRecognizer.create();

        eigenfaces.train(fotos, rotulos);
        eigenfaces.save("C:\\Projetos\\Reconhecimento_facial\\app\\src\\main\\java\\com\\resource\\treinamentoEigenfaces.yml");
        fisherfaces.train(fotos, rotulos);
        fisherfaces.save("C:\\Projetos\\Reconhecimento_facial\\app\\src\\main\\java\\com\\resource\\treinamentoFisherfaces.yml");
        lbph.train(fotos, rotulos);
        lbph.save("C:\\Projetos\\Reconhecimento_facial\\app\\src\\main\\java\\com\\resource\\treinamentoLBPH.yml");

        
    }
}