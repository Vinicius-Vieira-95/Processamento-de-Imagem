import cv2
import numpy as np

# =========================
# 1. Ler imagens
# =========================
img1 = cv2.imread('ima-A.png')
img2 = cv2.imread('img-B.png')

if img1 is None or img2 is None:
    raise ValueError("Erro ao carregar imagens")

# =========================
# 2. Converter para cinza (manual)
# =========================
def rgb_to_gray(image):
    h, w, _ = image.shape
    gray = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            r, g, b = image[i, j]
            gray[i, j] = 0.299*r + 0.587*g + 0.114*b

    return gray

gray1 = rgb_to_gray(img1)
gray2 = rgb_to_gray(img2)

# =========================
# 3. Garantir mesmo tamanho
# =========================
if gray1.shape != gray2.shape:
    raise ValueError("As imagens devem ter o mesmo tamanho")

# =========================
# 4. Combinação ponderada
# =========================
def blend_images(img1, img2, alpha):
    h, w = img1.shape
    result = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            result[i, j] = alpha * img1[i, j] + (1 - alpha) * img2[i, j]

    return np.clip(result, 0, 255).astype(np.uint8)

# =========================
# 5. Testes com diferentes pesos
# =========================
blend_03 = blend_images(gray1, gray2, 0.3)
blend_05 = blend_images(gray1, gray2, 0.5)
blend_07 = blend_images(gray1, gray2, 0.7)

# =========================
# 6. Salvar resultados
# =========================
cv2.imwrite('resp3/blend_03.png', blend_03)
cv2.imwrite('resp3/blend_05.png', blend_05)
cv2.imwrite('resp3/blend_07.png', blend_07)

print("Combinação concluída!")