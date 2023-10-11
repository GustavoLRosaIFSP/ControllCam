
import pytesseract
import cv2

import cv2
import pytesseract

def detecta_carro(source):
    cascade_src = 'Detectar Carro/Main Project/Main Project/Car Detection/cars.xml'
    car_cascade = cv2.CascadeClassifier(cascade_src)

    cap = cv2.VideoCapture(source)

    num_consecutive_frames = 100
    consecutive_frame_count = 0
    total_time_seconds = 0

    while True:
        ret, img = cap.read()
        
        if not ret:
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 2)

        if len(cars) > 0:
            consecutive_frame_count += 1
        else:
            consecutive_frame_count = 0

        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow('video', img)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_time_seconds = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        
        if consecutive_frame_count >= num_consecutive_frames:
            print("Carro detectado!")
            print("Tempo total do vídeo até a detecção: {:.2f} segundos".format(total_time_seconds))
            break

        if cv2.waitKey(33) == 27 and 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return total_time_seconds

def desenhaContornos(contornos, imagem):
    for c in contornos:
        # perimetro do contorno, verifica se o contorno é fechado
        perimetro = cv2.arcLength(c, True)
        if perimetro > 120:
            # aproxima os contornos da forma correspondente
            approx = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            # verifica se é um quadrado ou retangulo de acordo com a qtd de vertices
            if len(approx) == 4:
                # Contorna a placa atraves dos contornos encontrados
                (x, y, lar, alt) = cv2.boundingRect(c)
                cv2.rectangle(imagem, (x, y), (x + lar, y + alt), (0, 255, 0), 2)
                # segmenta a placa da imagem
                roi = imagem[y:y + alt, x:x + lar]
                cv2.imwrite(r'C:\Users\cpuG\Pictures\Placas_De_Carro\roi.png', roi)


def buscaRetanguloPlaca(source):
    # Captura ou Video
    video = cv2.VideoCapture(source)

    while video.isOpened():

        ret, frame = video.read()

        if (ret == False):
            break

        # area de localização u 720p
        area = frame[500:, 300:800]

        # area de localização 480p
        # area = frame[350:, 220:500]

        # escala de cinza
        img_result = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

        # limiarização
        ret, img_result = cv2.threshold(img_result, 90, 255, cv2.THRESH_BINARY)

        # desfoque
        img_result = cv2.GaussianBlur(img_result, (5, 5), 0)

        # lista os contornos
        contornos, hier = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # limite horizontal
        cv2.line(frame, (0, 500), (1280, 500), (0, 0, 255), 1)
        # limite vertical 1
        cv2.line(frame, (300, 0), (300, 720), (0, 0, 255), 1)
        # limite vertical 2
        cv2.line(frame, (800, 0), (800, 720), (0, 0, 255), 1)

        cv2.imshow('FRAME', frame)

        desenhaContornos(contornos, area)

        cv2.imshow('RES', area)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    video.release()
    preProcessamentoRoi()
    cv2.destroyAllWindows()


def preProcessamentoRoi():
    img_roi = cv2.imread(r'C:\Users\cpuG\Pictures\Placas_De_Carro\roi.png')
    # cv2.imshow("ENTRADA", img_roi)
    if img_roi is None:
        return

    # redmensiona a imagem da placa em 4x
    img = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Converte para escala de cinza
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Escala Cinza", img)

    # Binariza imagem
    _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    cv2.imshow("Limiar", img)

    # Desfoque na Imagem
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow("Desfoque", img)

    # Aplica reconhecimento OCR no ROI com o Tesseract
    cv2.imwrite('Resource/roi.png', img)

    return img


def reconhecimentoOCR():
    img_roi_ocr = cv2.imread('Resource/roi.png')
    if img_roi_ocr is None:
        return

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ013456789 --psm 6'
    saida = pytesseract.image_to_string(img_roi_ocr, lang='eng', config=config)


    print(saida.splitlines()[0])
    return saida.splitlines()[0]


if __name__ == "__main__":
    source = 'Resource/Arquivo_Camera_Principal.mkv'
    
    # Chama a primeira parte do código
    tempo_detectado = detecta_carro(source)
    
    # Executa a segunda parte do código apenas se a primeira parte for bem-sucedida
    if tempo_detectado > 0:
        buscaRetanguloPlaca(source)
        preProcessamentoRoi()
        reconhecimentoOCR()
        