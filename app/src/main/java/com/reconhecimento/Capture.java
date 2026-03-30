package com.reconhecimento;

import java.awt.event.KeyEvent;
import java.util.logging.Level;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class Capture {
    
    private static final java.util.logging.Logger logger = java.util.logging.Logger.getLogger(Capture.class.getName());
    
    public static void main(String[] args) {
        try {
            Contexto contexto = criarContexto();
            obterIdUsuario(contexto);
            inicializarCamera(contexto);
            inicializarDetector(contexto);
            inicializarJanela(contexto);
            iniciarCaptura(contexto);
            finalizarCaptura(contexto);
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Ocorreu um erro inesperado: " + e.getMessage(), e);
        }
    }

    
    // Métodos refatorados para melhorar a legibilidade e organização do código


    // Metodo para as variáveis necessárias para a captura de vídeo e processamento de imagens
    public static class Contexto{
        int idUsuario;
        OpenCVFrameGrabber camera;
        CascadeClassifier detector;
        CanvasFrame janela;
        OpenCVFrameConverter.ToMat conversor;
        int numeroAmostras;
        int amostrasColetadas;
        KeyEvent tecla;
        RectVector faces;
        Frame frameAtual;
        Mat imagemColorida;
        Mat imagemCinza;
        boolean executando;
    }
    
    private static Contexto criarContexto() {
        Contexto contexto = new Contexto();
        contexto.idUsuario = 0;
        contexto.camera = null;
        contexto.detector = null;
        contexto.janela = null;
        contexto.conversor = new OpenCVFrameConverter.ToMat();
        contexto.numeroAmostras = 25;
        contexto.amostrasColetadas = 0;
        contexto.tecla = null;
        contexto.faces = new RectVector();
        contexto.frameAtual = null;
        contexto.imagemColorida = new Mat();
        contexto.imagemCinza = new Mat();
        contexto.executando = true;
        
        return contexto;
    }
    
    // Método para obter o ID do usuário a partir do console
    private static void obterIdUsuario(Contexto contexto) {
        java.util.Scanner scanner = new java.util.Scanner(System.in);
        contexto.idUsuario = -1; // valor inválido para entrar no loop
        while (contexto.idUsuario <= 0) {
            try {
                logger.log(Level.INFO, "Digite um ID válido. Por favor, digite um número inteiro positivo.");
                contexto.idUsuario = scanner.nextInt();
                if (contexto.idUsuario <= 0) {
                    logger.log(Level.WARNING, "ID do usuário deve ser um número inteiro positivo. Tente novamente.");
                }
            } catch (java.util.InputMismatchException e) {
                logger.log(Level.SEVERE, "Entrada inválida. Por favor, digite um número inteiro positivo.");
                scanner.nextLine(); // Limpar o buffer do scanner para evitar loop infinito
            }
        }
        // Nao usar o "scanner.close();" para fechar o programa pois pode causar problemas e não ler outras entradas no futuro
    }

    // Método para inicializar a câmera e configurar o FrameGrabber
    private static void inicializarCamera(Contexto contexto) {
        contexto.camera = new OpenCVFrameGrabber(0);
        try {
            contexto.camera.start();
        } catch (org.bytedeco.javacv.FrameGrabber.Exception e) {
            logger.log(Level.SEVERE, "Erro ao iniciar a câmera: " + e.getMessage());
            System.exit(1);
        }
        // se houver mais de uma câmera, pode ser necessário ajustar o índice do FrameGrabber para acessar a câmera correta, fazer depois.
    }

    // Método para inicializar o detector de faces usando o CascadeClassifier
    private static void inicializarDetector(Contexto contexto) {
        contexto.detector = new CascadeClassifier("src/main/java/com/resource/haarcascade_frontalface_alt.xml");
        if (contexto.detector.empty()) {
            logger.log(Level.SEVERE, "Erro ao carregar o classificador Haar Cascade. Verifique o caminho do arquivo.");
            System.exit(1);
        }
    }

    // Método para iniciar o loop de captura de vídeo, detectar faces e exibir o resultado em uma janela
    private static void inicializarJanela(Contexto contexto) {
        contexto.janela = new CanvasFrame("Reconhecimento Facial", CanvasFrame.getDefaultGamma() / contexto.camera.getGamma());
        contexto.janela.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        contexto.janela.setVisible(true);
    }

    // Método para processar cada frame capturado, detectar faces, desenhar retângulos, exibir a janela, e coletar amostras de rosto para treinamento
    private static void iniciarCaptura(Contexto contexto) {
        
    }

    // Método para liberar recursos e fechar a janela ao finalizar a captura
    private static void finalizarCaptura(Contexto contexto) {
        contexto.janela.dispose();
        try {
            contexto.camera.stop();
        } catch (org.bytedeco.javacv.FrameGrabber.Exception e) {
            logger.log(Level.SEVERE, "Erro ao parar a câmera: " + e.getMessage());
        }
    }
}