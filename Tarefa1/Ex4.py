import cv2
import numpy as np

# =========================
# 1. Ler imagem
# =========================
img = cv2.imread('resp4/A.png')

if img is None:
    raise ValueError("Erro ao carregar imagem")

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

gray = rgb_to_gray(img)

# =========================
# 3. NEGATIVO
# =========================
def negativo(image):
    h, w = image.shape
    result = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            result[i, j] = 255 - image[i, j]

    return result

neg = negativo(gray)

# =========================
# 4. INTERVALO [100,200]
# =========================
def ajuste_intervalo(image, new_min=100, new_max=200):
    old_min = np.min(image)
    old_max = np.max(image)

    h, w = image.shape
    result = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            result[i, j] = ((image[i, j] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    return result

ajustado = ajuste_intervalo(gray)

# =========================
# 5. INVERTER LINHAS PARES
# =========================
def inverter_linhas_pares(image):
    h, w = image.shape
    result = image.copy()

    for i in range(0, h, 2):  # linhas pares
        result[i] = result[i][::-1]

    return result

linhas_invertidas = inverter_linhas_pares(gray)

# =========================
# 6. ESPELHAR METADE SUPERIOR
# =========================
def espelhar_superior(image):
    h, w = image.shape
    result = image.copy()

    metade = h // 2

    for i in range(metade):
        result[h - i - 1] = image[i]

    return result

espelhado_top = espelhar_superior(gray)

# =========================
# 7. ESPELHAMENTO VERTICAL COMPLETO
# =========================
def espelhamento_vertical(image):
    h, w = image.shape
    result = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            result[i, j] = image[i, w - j - 1]

    return result

espelho_vertical = espelhamento_vertical(gray)

# =========================
# 8. Salvar resultados
# =========================
cv2.imwrite('resp4/B.png', neg)
cv2.imwrite('resp4/C.png', ajustado.astype(np.uint8))
cv2.imwrite('resp4/D.png', linhas_invertidas.astype(np.uint8))
cv2.imwrite('resp4/E.png', espelhado_top.astype(np.uint8))
cv2.imwrite('resp4/F.png', espelho_vertical.astype(np.uint8))

print("Concluído!")