import cv2
import numpy as np

# =========================
# 1. Ler imagem
# =========================
img = cv2.imread('arte-3.png')

if img is None:
    raise ValueError("Erro ao carregar a imagem.")

# =========================
# 2. Converter para cinza manualmente
# =========================
def rgb_to_gray(image):
    h, w, _ = image.shape
    gray = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            b, g, r = image[i, j]  # OpenCV lê em BGR
            gray[i, j] = 0.299 * r + 0.587 * g + 0.114 * b

    return gray

gray = rgb_to_gray(img)

# =========================
# 3. Função de quantização
# =========================
def quantizar(image, niveis):
    h, w = image.shape
    result = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            pixel = image[i, j]

            # reduz para o número de níveis
            nivel = int((pixel * niveis) / 256)

            if nivel >= niveis:
                nivel = niveis - 1

            # expande de volta para [0,255]
            if niveis > 1:
                result[i, j] = (nivel * 255) / (niveis - 1)
            else:
                result[i, j] = 0

    return np.clip(result, 0, 255).astype(np.uint8)

# =========================
# 4. Aplicar diferentes quantizações
# =========================
q256 = quantizar(gray, 256)
q64  = quantizar(gray, 64)
q32  = quantizar(gray, 32)
q16  = quantizar(gray, 16)
q8   = quantizar(gray, 8)
q4   = quantizar(gray, 4)
q2   = quantizar(gray, 2)

# =========================
# 5. Salvar resultados
# =========================
cv2.imwrite('resp6/quantizacao_256.png', q256)
cv2.imwrite('resp6/quantizacao_64.png', q64)
cv2.imwrite('resp6/quantizacao_32.png', q32)
cv2.imwrite('resp6/quantizacao_16.png', q16)
cv2.imwrite('resp6/quantizacao_8.png', q8)
cv2.imwrite('resp6/quantizacao_4.png', q4)
cv2.imwrite('resp6/quantizacao_2.png', q2)

print("Concluído.")