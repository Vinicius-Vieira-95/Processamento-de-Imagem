import cv2
import numpy as np

# =========================
# 1. Ler imagem
# =========================
img = cv2.imread('arte-3.png')

if img is None:
    raise ValueError("Não foi possível carregar a imagem.")

# converter para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# 2. Função de correção gama
# =========================
def gamma_correction(image, gamma):
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            # normaliza para [0,1]
            a = image[i, j] / 255.0

            # aplica B = A^(1/gamma)
            b = a ** (1.0 / gamma)

            # volta para [0,255]
            output[i, j] = b * 255.0

    return np.clip(output, 0, 255).astype(np.uint8)

# =========================
# 3. Testar diferentes gamas
# =========================
gamma_05 = gamma_correction(gray, 0.5)
gamma_10 = gamma_correction(gray, 1.5)
gamma_20 = gamma_correction(gray, 2.5)
gamma_30 = gamma_correction(gray, 3.5)

# =========================
# 4. Salvar resultados
# =========================
cv2.imwrite('resp2/gray_original.png', gray)
cv2.imwrite('resp2/gamma_0_5.png', gamma_05)
cv2.imwrite('resp2/gamma_1_5.png', gamma_10)
cv2.imwrite('resp2/gamma_2_5.png', gamma_20)
cv2.imwrite('resp2/gamma_3_5.png', gamma_30)

print("Imagens geradas com sucesso.")