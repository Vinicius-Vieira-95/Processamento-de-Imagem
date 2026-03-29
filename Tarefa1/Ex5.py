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
            b, g, r = image[i, j]  # cv2 lê em BGR
            gray[i, j] = 0.299 * r + 0.587 * g + 0.114 * b

    return gray

gray = rgb_to_gray(img).astype(np.uint8)

# =========================
# 3. Garantir divisibilidade por 4
# =========================
h, w = gray.shape

novo_h = (h // 4) * 4
novo_w = (w // 4) * 4

gray = gray[:novo_h, :novo_w]

h, w = gray.shape
bloco_h = h // 4
bloco_w = w // 4

# =========================
# 4. Separar os 16 blocos
# =========================
blocos = []

for i in range(4):
    for j in range(4):
        inicio_h = i * bloco_h
        fim_h = inicio_h + bloco_h
        inicio_w = j * bloco_w
        fim_w = inicio_w + bloco_w

        bloco = gray[inicio_h:fim_h, inicio_w:fim_w]
        blocos.append(bloco)

# =========================
# 5. Definir nova ordem
# =========================
# Exemplo:
# ordem = [15, 14, 13, 12,
#          11, 10,  9,  8,
#           7,  6,  5,  4,
#           3,  2,  1,  0]
#
# Ajuste esta lista conforme a figura do enunciado.

ordem = [
    15, 14, 13, 12,
    11, 10,  9,  8,
     7,  6,  5,  4,
     3,  2,  1,  0
]

# =========================
# 6. Montar imagem final
# =========================
resultado = np.zeros((h, w), dtype=np.uint8)

indice = 0
for i in range(4):
    for j in range(4):
        inicio_h = i * bloco_h
        fim_h = inicio_h + bloco_h
        inicio_w = j * bloco_w
        fim_w = inicio_w + bloco_w

        resultado[inicio_h:fim_h, inicio_w:fim_w] = blocos[ordem[indice]]
        indice += 1

# =========================
# 7. Salvar
# =========================
cv2.imwrite('resp5/mosaico_4x4.png', resultado)

print("Questão 5 concluída com sucesso.")