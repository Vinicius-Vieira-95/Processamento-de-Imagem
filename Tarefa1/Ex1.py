
import cv2
import numpy as np

# =========================
#  Ler imagem
# =========================
# (i) Converter a imagem colorida para níveis de cinza,
img = cv2.imread('arte-2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb_to_gray(image):
    h, w, _ = image.shape
    gray = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            r, g, b = image[i, j]
            gray[i, j] = 0.300*r + 0.500*g + 0.100*b # cada cor indo para o tom cinza

    return gray
gray = rgb_to_gray(img)

# =========================
#  kernel gaussiano
# =========================
def gaussian_kernel(size=5, sigma=1):
    k = size // 2
    kernel = np.zeros((size, size))

    for x in range(-k, k+1):
        for y in range(-k, k+1):
            kernel[x+k, y+k] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))

    kernel = kernel / np.sum(kernel)
    return kernel

kernel = gaussian_kernel(5, 1)

# =========================
# Convolução manual
# suavisando a imagem neste processo
# =========================
def convolve(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2
    output = np.zeros_like(image)

    padded = np.pad(image, k, mode='edge')

    for i in range(h):
        for j in range(w):
            region = padded[i:i+2*k+1, j:j+2*k+1]
            output[i, j] = np.sum(region * kernel)

    return output

blur = convolve(gray, kernel)

# =========================
# 5. Inverter blur
# =========================
invert_blur = 255 - blur

# =========================
# 6. Color Dodge (divisão)
# =========================
def dodge(front, back):
    result = np.zeros_like(front)

    for i in range(front.shape[0]):
        for j in range(front.shape[1]):
            if back[i, j] == 255:
                result[i, j] = 255
            else:
                result[i, j] = min(255, (front[i, j] * 255) / (255 - back[i, j]))

    return result

sketch = dodge(gray, invert_blur)

# =========================
# 7. Salvar resultado
# =========================
sketch = np.clip(sketch, 0, 255).astype(np.uint8)
cv2.imwrite('resp1/resultado_lapis.png', sketch)

print("Imagem gerada com sucesso!")