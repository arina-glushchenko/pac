import os
import cv2
import numpy as np


# функция для создания двухцветной карты с результатом
def create_motion_map(frame, motion_detected):
    # создаем чистый холст нужного размера
    motion_map = np.zeros(frame.shape[:2], dtype=np.uint8)
    # выбираем цвет в зависимости от того, обнаружено движение или нет
    color = (0, 0, 255) if motion_detected else (0, 255, 0)
    # отрисовываем прямоугольник на холсте
    cv2.rectangle(motion_map, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
    # смешиваем холст с кадром, чтобы получить двухцветную карту
    alpha = 0.3
    return cv2.addWeighted(frame, 1 - alpha, cv2.cvtColor(motion_map, cv2.COLOR_GRAY2BGR), alpha, 0)


# получаем доступ к камере
cap = cv2.VideoCapture(0)

# инициализируем переменные для обнаружения движения
prev_frame = None
motion_detected = False

# запускаем бесконечный цикл для чтения кадров из видеопотока
while True:
    # читаем текущий кадр из видеопотока
    ret, frame = cap.read()

    # если кадр не был прочитан, выходим из цикла
    if not ret:
        break

    # преобразуем текущий кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # если это не первый кадр, сравниваем его с предыдущим
    if prev_frame is not None:
        # вычисляем абсолютную разницу между текущим и предыдущим кадрами
        frame_diff = cv2.absdiff(prev_frame, gray)
        # применяем пороговую фильтрацию для выделения областей с движением
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        # вычисляем контуры объектов на изображении
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # если найден хотя бы один контур, считаем, что движение обнаружено
        motion_detected = len(contours) > 0

    # сохраняем текущий кадр для использования на следующей итерации
    prev_frame = gray

    # создаем двухцветную карту с результатом
    motion_map = create_motion_map(frame, motion_detected)

    # отображаем двухцветную карту в окне
    cv2.imshow('Motion Map', motion_map)

    # выводим текстовое сообщение о текущем режиме программы
    status_text = 'Red Light' if motion_detected else 'Green Light'
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if motion_detected else (0, 255, 0), 2)

    # находим контуры объектов на кадре
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # рисуем контуры объектов красным цветом, если обнаружено движение
    if motion_detected:
        frame = cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    # ждем нажатия клавиши
    key = cv2.waitKey(1) & 0xFF

    # если нажата клавиша "q", выходим из цикла
    if key == ord('q'):
        break

# освобождаем ресурсы
cv2.destroyAllWindows()
cap.release()