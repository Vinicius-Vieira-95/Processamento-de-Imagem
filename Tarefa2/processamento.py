"""
Atividade 2 de Processamento de Imagens
Imagem-base: 'Comedian' (Maurizio Cattelan) — banana fixada com fita adesiva.

Implementação manual de:
1) Filtros espaciais h1 a h11 (convolução)
2) Filtros no domínio da frequência (FFT 2D, passa-baixa, passa-alta,
   passa-faixa, rejeita-faixa) e compressão por limiarização.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

IMG_IN  = "/mnt/user-data/uploads/arte-banana.png"
OUT_DIR = "/home/claude/trabalho_pi/saidas"
os.makedirs(OUT_DIR, exist_ok=True)

# Carrega em escala de cinza (OpenCV apenas para abrir/salvar)
img = cv2.imread(IMG_IN, cv2.IMREAD_GRAYSCALE)
print("Imagem carregada:", img.shape, img.dtype, "min/max:", img.min(), img.max())
cv2.imwrite(os.path.join(OUT_DIR, "00_imagem_original.png"), img)


# ======================================================================
# QUESTÃO 1 — FILTROS ESPACIAIS (CONVOLUÇÃO MANUAL)
# ======================================================================

def convolucao_2d(imagem, kernel):
    """Convolução 2D manual com inversão do kernel e padding por replicação."""
    imagem = imagem.astype(np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kernel_flip = kernel[::-1, ::-1]
    kh, kw = kernel_flip.shape
    ph, pw = kh // 2, kw // 2
    img_pad = np.pad(imagem, ((ph, ph), (pw, pw)), mode="edge")
    H, W = imagem.shape
    saida = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            saida[i, j] = np.sum(img_pad[i:i + kh, j:j + kw] * kernel_flip)
    return saida


def normalizar_para_uint8(img_float, modo="clip"):
    if modo == "clip":
        return np.clip(img_float, 0, 255).astype(np.uint8)
    if modo == "abs":
        return np.clip(np.abs(img_float), 0, 255).astype(np.uint8)
    if modo == "shift":
        mn, mx = img_float.min(), img_float.max()
        if mx - mn < 1e-9:
            return np.zeros_like(img_float, dtype=np.uint8)
        return (((img_float - mn) / (mx - mn)) * 255).astype(np.uint8)
    raise ValueError(modo)


h1 = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0, -1, -2, -1,  0],
    [-1, -2, 16, -2, -1],
    [ 0, -1, -2, -1,  0],
    [ 0,  0, -1,  0,  0],
], dtype=np.float64)

h2 = (1 / 256.0) * np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1],
], dtype=np.float64)

h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
h4 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
h5 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
h6 = (1 / 9.0) * np.ones((3, 3), dtype=np.float64)
h7 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float64)
h8 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float64)
h9 = (1 / 9.0) * np.eye(9, dtype=np.float64)

h10 = (1 / 8.0) * np.array([
    [-1, -1, -1, -1, -1],
    [-1,  2,  2,  2, -1],
    [-1,  2,  8,  2, -1],
    [-1,  2,  2,  2, -1],
    [-1, -1, -1, -1, -1],
], dtype=np.float64)

h11 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float64)

filtros = [
    ("h1",  h1,  "abs"),
    ("h2",  h2,  "clip"),
    ("h3",  h3,  "abs"),
    ("h4",  h4,  "abs"),
    ("h5",  h5,  "abs"),
    ("h6",  h6,  "clip"),
    ("h7",  h7,  "abs"),
    ("h8",  h8,  "abs"),
    ("h9",  h9,  "clip"),
    ("h10", h10, "abs"),
    ("h11", h11, "shift"),
]


def aplicar_questao1(imagem):
    print("\n=== QUESTÃO 1: Filtros espaciais h1..h11 ===")
    resultados = {}
    for nome, k, modo in filtros:
        print(f"   aplicando {nome} ({k.shape[0]}x{k.shape[1]})...")
        r = convolucao_2d(imagem, k)
        r_u8 = normalizar_para_uint8(r, modo=modo)
        cv2.imwrite(os.path.join(OUT_DIR, f"q1_{nome}.png"), r_u8)
        resultados[nome] = r_u8
    return resultados


# ======================================================================
# QUESTÃO 2 — DOMÍNIO DA FREQUÊNCIA (FFT)
# ======================================================================

def fft2_manual(imagem):
    """FFT 2D via separabilidade (FFT 1D em linhas + colunas)."""
    f = imagem.astype(np.float64)
    f = np.fft.fft(f, axis=1)
    f = np.fft.fft(f, axis=0)
    return f


def ifft2_manual(F):
    g = np.fft.ifft(F, axis=0)
    g = np.fft.ifft(g, axis=1)
    return g


def fftshift_manual(F):
    """fftshift que funciona com qualquer dimensão (par ou ímpar)."""
    H, W = F.shape
    h_shift = H // 2
    w_shift = W // 2
    out = np.roll(F, shift=h_shift, axis=0)
    out = np.roll(out, shift=w_shift, axis=1)
    return out


def ifftshift_manual(F):
    """ifftshift inverso (rolagem em direção oposta para tratar dim ímpar)."""
    H, W = F.shape
    h_shift = -(H // 2)
    w_shift = -(W // 2)
    out = np.roll(F, shift=h_shift, axis=0)
    out = np.roll(out, shift=w_shift, axis=1)
    return out


# ---- máscaras circulares -------------------------------------------
def mascara_circular(shape, raio, dentro=1.0, fora=0.0):
    H, W = shape
    cy, cx = H // 2, W // 2
    y, x = np.mgrid[0:H, 0:W]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return np.where(dist <= raio, dentro, fora).astype(np.float64)


def mascara_passa_baixa(shape, raio):
    return mascara_circular(shape, raio, 1.0, 0.0)


def mascara_passa_alta(shape, raio):
    return mascara_circular(shape, raio, 0.0, 1.0)


def mascara_passa_faixa(shape, r_int, r_ext):
    H, W = shape
    cy, cx = H // 2, W // 2
    y, x = np.mgrid[0:H, 0:W]
    d = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return ((d >= r_int) & (d <= r_ext)).astype(np.float64)


def mascara_rejeita_faixa(shape, r_int, r_ext):
    return 1.0 - mascara_passa_faixa(shape, r_int, r_ext)


def aplicar_filtro_freq(imagem, mascara):
    F = fft2_manual(imagem)
    Fc = fftshift_manual(F)
    Ff = Fc * mascara
    F_back = ifftshift_manual(Ff)
    g = ifft2_manual(F_back)
    g = np.real(g)
    g = np.clip(g, 0, 255).astype(np.uint8)
    return g, Fc, Ff


def espectro_para_visualizacao(F):
    mag = np.log1p(np.abs(F))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)
    return (mag * 255).astype(np.uint8)


def aplicar_questao2_filtros(imagem):
    print("\n=== QUESTÃO 2 (parte A): Filtragem no domínio da frequência ===")

    F = fft2_manual(imagem)
    Fc = fftshift_manual(F)
    cv2.imwrite(os.path.join(OUT_DIR, "q2_espectro_original.png"),
                espectro_para_visualizacao(Fc))

    rec = np.real(ifft2_manual(F))
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, "q2_apos_inversa.png"), rec)
    erro = float(np.mean(np.abs(rec.astype(float) - imagem.astype(float))))
    print(f"   Erro médio absoluto FFT->IFFT (sanity check, ~0): {erro:.6f}")

    shape = imagem.shape
    # raios proporcionais ao tamanho da imagem (438x565)
    raios_pb = [15, 40, 80]
    raios_pa = [10, 25, 50]
    faixas   = [(15, 50), (30, 80)]

    saidas = {}

    for r in raios_pb:
        m = mascara_passa_baixa(shape, r)
        out, _, Ff = aplicar_filtro_freq(imagem, m)
        cv2.imwrite(os.path.join(OUT_DIR, f"q2_pb_r{r}.png"), out)
        cv2.imwrite(os.path.join(OUT_DIR, f"q2_pb_r{r}_mascara.png"), (m * 255).astype(np.uint8))
        saidas[f"pb_{r}"] = out

    for r in raios_pa:
        m = mascara_passa_alta(shape, r)
        out, _, Ff = aplicar_filtro_freq(imagem, m)
        cv2.imwrite(os.path.join(OUT_DIR, f"q2_pa_r{r}.png"), out)
        cv2.imwrite(os.path.join(OUT_DIR, f"q2_pa_r{r}_mascara.png"), (m * 255).astype(np.uint8))
        saidas[f"pa_{r}"] = out

    for (ri, re) in faixas:
        m_pf = mascara_passa_faixa(shape, ri, re)
        m_rf = mascara_rejeita_faixa(shape, ri, re)
        out_pf, _, _ = aplicar_filtro_freq(imagem, m_pf)
        out_rf, _, _ = aplicar_filtro_freq(imagem, m_rf)
        cv2.imwrite(os.path.join(OUT_DIR, f"q2_pf_{ri}_{re}.png"), out_pf)
        cv2.imwrite(os.path.join(OUT_DIR, f"q2_rf_{ri}_{re}.png"), out_rf)
        saidas[f"pf_{ri}_{re}"] = out_pf
        saidas[f"rf_{ri}_{re}"] = out_rf

    return saidas


# ---- compressão -----------------------------------------------------
def comprimir_por_limiar(imagem, percentual_descartado):
    F = fft2_manual(imagem)
    Fc = fftshift_manual(F)
    mag = np.abs(Fc)
    limiar = np.percentile(mag, percentual_descartado)
    mascara = (mag >= limiar).astype(np.float64)
    Ff = Fc * mascara
    F_back = ifftshift_manual(Ff)
    rec = np.real(ifft2_manual(F_back))
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    taxa = 1.0 - mascara.mean()
    return rec, taxa, Ff


def histograma_manual(imagem):
    h = np.zeros(256, dtype=np.int64)
    for v in imagem.ravel():
        h[v] += 1
    return h


def aplicar_questao2_compressao(imagem):
    print("\n=== QUESTÃO 2 (parte B): Compressão no domínio de frequência ===")
    perc_list = [50, 80, 95, 99]
    resultados = {}
    h_orig = histograma_manual(imagem)

    for p in perc_list:
        rec, taxa, _ = comprimir_por_limiar(imagem, p)
        cv2.imwrite(os.path.join(OUT_DIR, f"q2_compressao_p{p}.png"), rec)
        erro = float(np.mean((imagem.astype(float) - rec.astype(float)) ** 2))
        psnr = 10 * math.log10((255.0 ** 2) / max(erro, 1e-12))
        print(f"   p={p}%  coef. zerados={taxa*100:5.2f}%  MSE={erro:7.2f}  PSNR={psnr:5.2f} dB")
        resultados[p] = (rec, taxa, erro, psnr)

    return resultados, h_orig


# ----------------------------------------------------------------------
# FIGURAS COMPARATIVAS
# ----------------------------------------------------------------------
def gerar_figura_q1(imagem, resultados_q1):
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.ravel()
    axes[0].imshow(imagem, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i, (nome, _, _) in enumerate(filtros, start=1):
        axes[i].imshow(resultados_q1[nome], cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(nome)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_q1_comparativo.png"), dpi=130, bbox_inches="tight")
    plt.close()


def gerar_figura_q1_destaque_bordas(imagem, resultados_q1):
    """Destaca os detectores de borda (Sobel + Laplaciano) lado a lado."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(imagem, cmap="gray", vmin=0, vmax=255); axes[0].set_title("Original")
    axes[1].imshow(resultados_q1["h3"], cmap="gray"); axes[1].set_title("h3 — Sobel x (bordas verticais)")
    axes[2].imshow(resultados_q1["h4"], cmap="gray"); axes[2].set_title("h4 — Sobel y (bordas horizontais)")
    axes[3].imshow(resultados_q1["h5"], cmap="gray"); axes[3].set_title("h5 — Laplaciano")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_q1_bordas.png"), dpi=130, bbox_inches="tight")
    plt.close()


def gerar_figura_q1_suavizacao(imagem, resultados_q1):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(imagem, cmap="gray", vmin=0, vmax=255); axes[0].set_title("Original")
    axes[1].imshow(resultados_q1["h2"], cmap="gray", vmin=0, vmax=255); axes[1].set_title("h2 — Gaussiano 5x5")
    axes[2].imshow(resultados_q1["h6"], cmap="gray", vmin=0, vmax=255); axes[2].set_title("h6 — Média 3x3")
    axes[3].imshow(resultados_q1["h9"], cmap="gray", vmin=0, vmax=255); axes[3].set_title("h9 — Motion blur 9x9")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_q1_suavizacao.png"), dpi=130, bbox_inches="tight")
    plt.close()


def gerar_figura_q1_realce(imagem, resultados_q1):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(imagem, cmap="gray", vmin=0, vmax=255); axes[0].set_title("Original")
    axes[1].imshow(resultados_q1["h1"], cmap="gray"); axes[1].set_title("h1 — Realce 5x5 (LoG-like)")
    axes[2].imshow(resultados_q1["h10"], cmap="gray"); axes[2].set_title("h10 — Realce 5x5")
    axes[3].imshow(resultados_q1["h11"], cmap="gray"); axes[3].set_title("h11 — Emboss")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_q1_realce.png"), dpi=130, bbox_inches="tight")
    plt.close()


def gerar_figura_q2_filtros(imagem, saidas):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()
    titulos = [
        ("Original",                imagem),
        ("Passa-baixa r=15",        saidas["pb_15"]),
        ("Passa-baixa r=80",        saidas["pb_80"]),
        ("Passa-alta r=10",         saidas["pa_10"]),
        ("Passa-alta r=50",         saidas["pa_50"]),
        ("Passa-faixa 15-50",       saidas["pf_15_50"]),
        ("Passa-faixa 30-80",       saidas["pf_30_80"]),
        ("Rejeita-faixa 15-50",     saidas["rf_15_50"]),
        ("Rejeita-faixa 30-80",     saidas["rf_30_80"]),
    ]
    for ax, (t, im) in zip(axes, titulos):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_q2_filtros.png"), dpi=130, bbox_inches="tight")
    plt.close()


def gerar_figura_q2_mascaras(imagem):
    shape = imagem.shape
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(mascara_passa_baixa(shape, 40), cmap="gray")
    axes[0].set_title("Passa-baixa (r=40)")
    axes[1].imshow(mascara_passa_alta(shape, 25), cmap="gray")
    axes[1].set_title("Passa-alta (r=25)")
    axes[2].imshow(mascara_passa_faixa(shape, 15, 50), cmap="gray")
    axes[2].set_title("Passa-faixa (15-50)")
    axes[3].imshow(mascara_rejeita_faixa(shape, 15, 50), cmap="gray")
    axes[3].set_title("Rejeita-faixa (15-50)")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_q2_mascaras.png"), dpi=130, bbox_inches="tight")
    plt.close()


def gerar_figura_q2_compressao(imagem, resultados_compressao, h_orig):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(imagem, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    cols = sorted(resultados_compressao.keys())
    for i, p in enumerate(cols[:3], start=1):
        rec, taxa, mse, psnr = resultados_compressao[p]
        axes[0, i].imshow(rec, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"Comp. p={p}%\nzerados={taxa*100:.1f}%  PSNR={psnr:.1f}dB", fontsize=10)
        axes[0, i].axis("off")

    axes[1, 0].bar(np.arange(256), h_orig, color="blue", width=1.0)
    axes[1, 0].set_title("Histograma original")
    axes[1, 0].set_xlim([0, 255])
    for i, p in enumerate(cols[:3], start=1):
        rec, _, _, _ = resultados_compressao[p]
        h = histograma_manual(rec)
        axes[1, i].bar(np.arange(256), h, color="blue", width=1.0)
        axes[1, i].set_title(f"Histograma p={p}%")
        axes[1, i].set_xlim([0, 255])

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_q2_compressao.png"), dpi=130, bbox_inches="tight")
    plt.close()


def gerar_figura_espectro(imagem):
    F = fft2_manual(imagem)
    Fc = fftshift_manual(F)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(imagem, cmap="gray")
    axes[0].set_title("Imagem original")
    axes[0].axis("off")
    axes[1].imshow(espectro_para_visualizacao(F), cmap="gray")
    axes[1].set_title("Espectro |F| (sem shift)")
    axes[1].axis("off")
    axes[2].imshow(espectro_para_visualizacao(Fc), cmap="gray")
    axes[2].set_title("Espectro centralizado")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_q2_espectro.png"), dpi=130, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# EXECUÇÃO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    res_q1 = aplicar_questao1(img)
    gerar_figura_q1(img, res_q1)
    gerar_figura_q1_destaque_bordas(img, res_q1)
    gerar_figura_q1_suavizacao(img, res_q1)
    gerar_figura_q1_realce(img, res_q1)

    gerar_figura_espectro(img)
    saidas_q2 = aplicar_questao2_filtros(img)
    gerar_figura_q2_filtros(img, saidas_q2)
    gerar_figura_q2_mascaras(img)

    res_comp, h_orig = aplicar_questao2_compressao(img)
    gerar_figura_q2_compressao(img, res_comp, h_orig)

    print("\nProcessamento finalizado. Arquivos em:", OUT_DIR)
