
import pytesseract
import cv2
import pytz
from datetime import datetime

def detectar_carro_Bicileta(source):
    cascade_src = 'Detectar Carro/Main Project/Main Project/Car Detection/cars.xml'
    car_cascade = cv2.CascadeClassifier(cascade_src)
    second_cascade_src = 'Detectar Carro/Main Project/Main Project/Bike Detection/two_wheeler.xml'
    bike_cascade = cv2.CascadeClassifier(second_cascade_src)

    cap = cv2.VideoCapture(source)

    num_consecutive_frames = 100
    consecutive_frame_count = 0
    tempo_total_segundos = 0

    while True:
        ret, img = cap.read()
        
        if not ret:
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 2)
        bikes = bike_cascade.detectMultiScale(gray, 1.19, 1)

        if len(cars) > 0 or len(bikes) > 0:
            consecutive_frame_count += 1
        else:
            consecutive_frame_count = 0

        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        for (x, y, w ,h) in bikes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,215),2)

        cv2.imshow('video', img)

        fps = cap.get(cv2.CAP_PROP_FPS)
        tempo_total_segundos = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        
        if consecutive_frame_count >= num_consecutive_frames:
            print("Veiculo detectado!")
            print("Tempo total do vídeo até a detecção: {:.2f} segundos".format(tempo_total_segundos))
            break

        if cv2.waitKey(33) == 27 and 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return tempo_total_segundos

def desenhaContornos(contornos, imagem):
    for c in contornos:
        
        perimetro = cv2.arcLength(c, True)
        if perimetro > 120:
            
            approx = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            
            if len(approx) == 4:
              
                (x, y, lar, alt) = cv2.boundingRect(c)
                cv2.rectangle(imagem, (x, y), (x + lar, y + alt), (0, 255, 0), 2)
                
                roi = imagem[y:y + alt, x:x + lar]
                cv2.imwrite('Resource/placa.png', roi)


def buscaRetanguloPlaca(source):
    
    video = cv2.VideoCapture(source)

    while video.isOpened():

        ret, frame = video.read()

        if (ret == False):
            break

       
        area = frame[500:, 300:800]

        img_result = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

        ret, img_result = cv2.threshold(img_result, 90, 255, cv2.THRESH_BINARY)

        img_result = cv2.GaussianBlur(img_result, (5, 5), 0)

        contornos, hier = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cv2.line(frame, (0, 500), (1280, 500), (0, 0, 255), 1)

        cv2.line(frame, (300, 0), (300, 720), (0, 0, 255), 1)

        cv2.line(frame, (800, 0), (800, 720), (0, 0, 255), 1)

        cv2.imshow('FRAME', frame)

        desenhaContornos(contornos, area)

        cv2.imshow('RES', area)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def preProcessamentoRoi():
    img_roi = cv2.imread('Resource/placa.png')
    
    if img_roi is None:
        return

    img = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    cv2.imshow("Limiar", img)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite('Resource/placa_processada.png', img)

    return img

def reconhecimentoOCR():
    img_roi_ocr = cv2.imread('Resource/placa_processada.png')
    if img_roi_ocr is None:
        return

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ013456789 --psm 6'
    saida = pytesseract.image_to_string(img_roi_ocr, lang='eng', config=config)


    return saida.splitlines()[0]

def hora_atual():
    brasilia_tz = pytz.timezone('America/Sao_Paulo')
    horario_brasilia = datetime.now(brasilia_tz)
    horario_formatado = horario_brasilia.strftime('%Y-%m-%d %H:%M:%S %Z%z')
    return horario_formatado

if __name__ == "__main__":
    source = 'Resource/Arquivo_Camera_Principal.mkv'
    
    tempo_detectado = detectar_carro_Bicileta(source)
    
    if tempo_detectado > 0:
        buscaRetanguloPlaca(source)
        preProcessamentoRoi()
        placa_final = reconhecimentoOCR()
        hora_leitura = hora_atual()
        print(placa_final, '\n', hora_leitura, '\n')
        
