
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image                       # PIL: APENAS carregar/salvar

IMG_Q1 = "imagens/img_1.png"           # Questao 1 (compressao)
IMG_A  = "imagens/img_1.png"        # Questao 2 - muito detalhe
IMG_B  = "imagens/img_1.png"            # Questao 2 - homogenea
SAIDA  = "saidas"

# ============================ E/S ============================
def carregar(caminho):
    return np.asarray(Image.open(caminho), dtype=np.float64)

def para_cinza(arr):
    """RGB -> cinza MANUAL (luminancia Rec.601): Y=0.299R+0.587G+0.114B."""
    if arr.ndim == 2:
        return arr.copy()
    a = arr[..., :3]
    return 0.299*a[..., 0] + 0.587*a[..., 1] + 0.114*a[..., 2]

def salvar_cinza(arr, caminho):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L").save(caminho)

def _sintetica(tipo, M=256, N=256):
    yy, xx = np.mgrid[0:M, 0:N]
    if tipo == "textura":
        img = 110 + 55*np.sin(xx/3.0)*np.cos(yy/3.0) + 25*np.sin(xx/1.3 + yy/1.7)
    elif tipo == "ceu":
        img = 60 + 0.25*yy + 8*np.sin(xx/30.0)
    else:
        ceu = 60 + 0.25*yy + 8*np.sin(xx/30.0)
        veg = 110 + 55*np.sin(xx/3.0)*np.cos(yy/3.0)
        img = np.where(yy < M*0.45, ceu, veg)
    return np.clip(img, 0, 255)

def carregar_cinza(caminho, tipo_fallback):
    if os.path.exists(caminho):
        print(f"[OK] {caminho}")
        return para_cinza(carregar(caminho))
    print(f"[AVISO] '{caminho}' nao encontrado -> imagem SINTETICA")
    return _sintetica(tipo_fallback)

# ===================== Q1: DCT / IDCT / QUANT =====================
def matriz_dct(N):
    T = np.zeros((N, N))
    for u in range(N):
        cu = np.sqrt(1.0/N) if u == 0 else np.sqrt(2.0/N)
        for x in range(N):
            T[u, x] = cu*np.cos((2*x + 1)*u*np.pi/(2.0*N))
    return T

def dct2d(b, T):  return T @ b @ T.T
def idct2d(c, T): return T.T @ c @ T

Q_LUM = np.array([
    [16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]], dtype=np.float64)

def escalar_quantizacao(Q, qualidade):
    qualidade = max(1, min(100, qualidade))
    s = 5000.0/qualidade if qualidade < 50 else 200.0 - 2.0*qualidade
    Qs = np.floor((Q*s + 50.0)/100.0)
    Qs[Qs == 0] = 1.0
    return Qs

def comprimir_descomprimir(cinza, Q, N=8):
    M, C = cinza.shape
    pb, pr = (-M) % N, (-C) % N
    img = np.pad(cinza, ((0, pb), (0, pr)), mode="edge")
    T = matriz_dct(N)
    recon = np.zeros_like(img, dtype=np.float64)
    nz = tot = 0
    for i in range(0, img.shape[0], N):
        for j in range(0, img.shape[1], N):
            bloco = img[i:i+N, j:j+N] - 128.0
            cq = np.round(dct2d(bloco, T) / Q)
            nz += np.count_nonzero(cq); tot += cq.size
            recon[i:i+N, j:j+N] = idct2d(cq*Q, T) + 128.0
    return np.clip(recon[:M, :C], 0, 255), 100.0*nz/tot

def mse(a, b):  return float(np.mean((a.astype(float) - b.astype(float))**2))
def psnr(a, b):
    e = mse(a, b);  return float("inf") if e == 0 else float(10*np.log10(255.0**2/e))

# ===================== Q2: DESCRITORES =====================
def descritores(cinza):
    f = cinza.astype(np.float64); M, N = f.shape; n = M*N
    media = f.sum()/n
    variancia = ((f - media)**2).sum()/n
    energia_t = (f**2).sum()
    dh = np.abs(f[:, 1:] - f[:, :-1]); dv = np.abs(f[1:, :] - f[:-1, :])
    return {"media": media, "variancia": variancia, "desvio_padrao": np.sqrt(variancia),
            "energia_total": energia_t, "energia_media": energia_t/n,
            "var_espacial_h": dh.sum()/dh.size, "var_espacial_v": dv.sum()/dv.size,
            "var_espacial_tot": (dh.sum() + dv.sum())/(dh.size + dv.size)}

# ============================ MAIN ============================
def main():
    os.makedirs(SAIDA, exist_ok=True)
    img_q1 = carregar_cinza(IMG_Q1, "mista")
    img_a  = carregar_cinza(IMG_A,  "textura")
    img_b  = carregar_cinza(IMG_B,  "ceu")

    # ---- Q1 ----
    print("\n=== Q1: Compressao DCT ===")
    qualidades = [90, 50, 10]; recs = {}
    for q in qualidades:
        rec, pct = comprimir_descomprimir(img_q1, escalar_quantizacao(Q_LUM, q))
        recs[q] = rec
        print(f"q={q:3d} | PSNR={psnr(img_q1,rec):6.2f} dB | MSE={mse(img_q1,rec):8.2f} | nao-nulos={pct:5.1f}%")

    fig, ax = plt.subplots(1, len(qualidades)+1, figsize=(4*(len(qualidades)+1), 4))
    ax[0].imshow(img_q1, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Original"); ax[0].axis("off")
    for k, q in enumerate(qualidades, 1):
        ax[k].imshow(recs[q], cmap="gray", vmin=0, vmax=255)
        ax[k].set_title(f"q={q} (PSNR {psnr(img_q1,recs[q]):.1f})"); ax[k].axis("off")
    plt.tight_layout(); plt.savefig(f"{SAIDA}/q1_comparacao.png", dpi=110, bbox_inches="tight"); plt.close()

    dif = np.abs(img_q1 - recs[10])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    ax[0].imshow(img_q1, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(recs[10], cmap="gray", vmin=0, vmax=255); ax[1].set_title("Reconstruida q=10"); ax[1].axis("off")
    m = ax[2].imshow(dif, cmap="inferno"); ax[2].set_title("|Diferenca|"); ax[2].axis("off")
    fig.colorbar(m, ax=ax[2], fraction=0.046); plt.tight_layout()
    plt.savefig(f"{SAIDA}/q1_diferenca.png", dpi=110, bbox_inches="tight"); plt.close()
    salvar_cinza(recs[10], f"{SAIDA}/reconstruida_q10.png")

    # ---- Q2 ----
    print("\n=== Q2: Descritores ===")
    da, db = descritores(img_a), descritores(img_b)
    print(f"{'descritor':<18}{'IMG_A(textura)':>18}{'IMG_B(homog.)':>18}")
    for k in da:
        print(f"{k:<18}{da[k]:>18.3f}{db[k]:>18.3f}")

    labels = ["variancia", "var.espacial", "energia media/100"]
    va = [da["variancia"], da["var_espacial_tot"], da["energia_media"]/100]
    vb = [db["variancia"], db["var_espacial_tot"], db["energia_media"]/100]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w/2, va, w, label="IMG_A (textura)")
    ax.bar(x + w/2, vb, w, label="IMG_B (homogenea)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend()
    ax.set_title("Comparacao de descritores"); plt.tight_layout()
    plt.savefig(f"{SAIDA}/q2_descritores.png", dpi=110, bbox_inches="tight"); plt.close()

    print(f"\nFiguras salvas em ./{SAIDA}/")

if __name__ == "__main__":
    main()
