import cv2
import numpy as np

def optimize_c_value(gray_frame, initial_c=4, target_area=1000, tolerance=200, max_iterations=10):
    """
    Çizgi alanına göre optimal C değerini bulur.
    """
    c_value = initial_c
    for _ in range(max_iterations):
        # Adaptive threshold işlemi
        binary_frame = cv2.adaptiveThreshold(
            gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, c_value
        )
        # Kontur bulma
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(max_contour)
            
            # Eğer alan hedefe yakınsa durdur
            if abs(contour_area - target_area) <= tolerance:
                break
            
            # Alan büyükse C'yi artır, küçükse azalt
            if contour_area < target_area:
                c_value -= 1
            else:
                c_value += 1
            
        # C değerini sınırlandır
        c_value = max(1, c_value)  # C negatif olmamalı
    
    return c_value


# Burada videonun size ını belirledik (genislik,yükseklik)
fixed_size = (848, 480)

# videoyu yakaladık
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # videoyu sabit boyuta resizeladık
    frame_resized = cv2.resize(frame, fixed_size)
    cv2.imshow('frame_resized', frame_resized)

    # Gri tonlamaya çeviriyoruz
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_frame', gray_frame)

    # Gaussian blur uygulayarak gürültüyü azaltıyoruz
    blurred_frame = cv2.GaussianBlur(gray_frame, (15, 15), cv2.BORDER_DEFAULT)
    cv2.imshow('blurred_frame', blurred_frame)

    # Dinamik C değeri optimizasyonu
    optimal_c = optimize_c_value(blurred_frame, initial_c=4, target_area=1500, tolerance=200)

    binary_frame = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, optimal_c)
    cv2.imshow('binary_frame',binary_frame)
    
    base = np.ones((5, 5), np.uint8)
    clean_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, base)  # gürültüyü temizliyor
    clean_frame = cv2.morphologyEx(clean_frame, cv2.MORPH_CLOSE, base)  # Çizgi bütünlüğünü korumaya yarıyo

    # Renk ayrımına dayalı siyah çizgi tespiti
    hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Siyah renk için HSV aralığı belirleniyor
    lower_black = np.array([0, 0, 0])  # Siyah için alt sınır
    upper_black = np.array([180, 255, 50])  # Siyah için üst sınır

    # Siyah renkleri tespit etmek
    mask = cv2.inRange(hsv_frame, lower_black, upper_black)
    cv2.imshow('mask', mask)

    # Siyah çizgi konturları
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # cizgiyi bulmak için kontur degeri fazla olanı secmek için bu ifi olusturduk
        belirgin_contour = max(contours, key=cv2.contourArea)

        # belirgin çizgimizin etrafına dikdörtgen çizmeye odaklı(takip için) satır oluşturduk
        if cv2.contourArea(belirgin_contour) > 100:  # Minimum alan eşiği

            # Çizginin etrafını belirginleştirmek için dikdörtgen çiziyoruz
            x, y, w, h = cv2.boundingRect(belirgin_contour)    #x , y koordinatları weight ve height değerlerini atadık
            frame_center = fixed_size[0] // 2
            line_center = x + w // 2

            # sağa veya sola dönmek için orta noktayı belirledik resim ve çizgi olmak üzere
            if line_center < frame_center - 50:
                direction = "Saga Don"
            elif line_center > frame_center + 50:
                direction = "Sola Don"
            else:
                direction = "Duz Git"  # Çizgi tam ortadaysa düz git komutunu ekliyoruz

            # Konturun etrafını çiziyoruz
            cv2.drawContours(frame_resized, [belirgin_contour], -1, (0, 255, 0), -1)  # Yeşil renkte çizgi

            # ekrandaki sag sol yazıları icin hem de çizgi kalınılığı vs.
            text_size = cv2.getTextSize(direction, cv2.FONT_ITALIC, 1, 2)[0]
            text_x = (frame_resized.shape[1] - text_size[0]) // 2  #
            text_y = (frame_resized.shape[0] // 15)

            cv2.putText(frame_resized, direction, (text_x, text_y), cv2.FONT_ITALIC, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # yaptıklarımızı ekrana bastık
    cv2.imshow('Renan Yeni Video', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()