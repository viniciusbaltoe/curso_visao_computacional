# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: Vinícius Breda Altoé e Lázaro Villela
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

###############################################################################
# Função para normalizar pontos
# Entrada: points (pontos da imagem a serem normalizados)
# Saída: norm_points (pontos normalizados)
#        T (matriz de normalização)

def normalize_points(points):
    # Calcula as médias das coordenadas x e y
    mean_x = np.mean(points[0])
    mean_y = np.mean(points[1])

    # Calcula as distâncias médias em relação às médias calculadas
    mean_dist = np.mean(np.sqrt((points[0] - mean_x)**2 + (points[1] - mean_y)**2))

    # Calcula o fator de escala
    scale = np.sqrt(2) / mean_dist

    # Cria a matriz de normalização
    T = np.array([[scale,       0, -scale * mean_x],
                  [    0,   scale, -scale * mean_y],
                  [    0,       0,               1]])

    # Aplica a normalização aos pontos
    homogeneous_points = np.vstack((points, np.ones(points.shape[1])))
    norm_points = np.dot(T, homogeneous_points)

    return norm_points, T

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)
def construct_A(pts1, pts2):
    num_points = pts1.shape[1]
    matrices = []
    for i in range(num_points):
        x,  y,  w  = pts1[0][i], pts1[1][i], pts1[2][i]
        x_, y_, w_ = pts2[0][i], pts2[1][i], pts2[2][i]

        A_partial = np.array([[   0,    0,    0, -w_*x, -w_*y, -w_*w,  y_*x,  y_*y,  y_*w],
                              [w_*x, w_*y, w_*w,     0,     0,     0, -x_*x, -x_*y, -x_*w]])
        matrices.append(A_partial)
    return np.concatenate(matrices)


def compute_normalized_dlt(pts1, pts2):
    """ Função do DLT Normalizado

    Recebe pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1
    Entrada:
        pts1,
        pts2
    Saída:
        H_matrix (matriz de homografia estimada)
    """

    # Add homogeneous coordinates
    pts1_ = pts1.T
    pts1_ = np.vstack((pts1_, np.ones(pts1_.shape[1])))
    pts2_ = pts2.T
    pts2_ = np.vstack((pts2_, np.ones(pts2_.shape[1])))

    # Compute matrix A
    A = construct_A(pts1_, pts2_)

    # Perform SVD(A) = U.S.Vt to estimate the homography
    u, s, vt = np.linalg.svd(A)

    # Reshape last column of V as the homography matrix
    H_matrix = vt[-1].reshape((3, 3))
    return H_matrix

# Função do RANSAC
# Entradas:
# pts1: pontos da primeira imagem
# pts2: pontos da segunda imagem
# dis_threshold: limiar de distância a ser usado no RANSAC
# N: número máximo de iterações (pode ser definido dentro da função e deve ser atualizado
#    dinamicamente de acordo com o número de inliers/outliers)
# Ninl: limiar de inliers desejado (pode ser ignorado ou não - fica como decisão de vocês)
# Saídas:
# H: homografia estimada
# pts1_in, pts2_in: conjunto de inliers dos pontos da primeira e segunda imagens


def RANSAC(pts1, pts2, dis_threshold, N, Ninl):
    num_points = len(pts1)

    if num_points < 4:
        raise ValueError("Não há pontos suficientes para estimar a homografia.")

    best_inliers = []
    best_num_inliers = 0

    for _ in range(N):
        # Seleciona aleatoriamente 4 pontos
        random_indices = np.random.choice(num_points, size=4, replace=False)
        sampled_pts1 = pts1[random_indices, :]
        sampled_pts2 = pts2[random_indices, :]

        # Estima a homografia usando DLT normalizado
        H_matrix = compute_normalized_dlt(sampled_pts1[:,0], sampled_pts2[:,0])

        # Transformando o NumPy array em listas
        pts1_list = pts1.tolist()
        pts1_array = np.array([point[0] + [1] for point in pts1_list])
        pts1_transposed = pts1_array.T

        pts2_list = pts2.tolist()
        pts2_array = np.array([point[0] + [1] for point in pts2_list])
        pts2_transposed = pts2_array.T

        # Multiplica pela matriz de homografia
        projected_pts_homogeneous = np.dot(H_matrix, pts1_transposed)

        # Normaliza pelos últimos coordenadas (dividindo pela última linha)
        projected_pts_normalized = (projected_pts_homogeneous / projected_pts_homogeneous[2, :]).T

        # Calcula as distâncias entre os pontos projetados normalizados e os pontos reais
        # distances = np.sqrt(np.sum((projected_pts_normalized[:, :2] - pts2_transposed[:, :2])**2, axis=1))
        distances = np.sqrt(np.sum((projected_pts_normalized[:, :2] - pts2_transposed[:2, :].T)**2, axis=1))

        # Identifica inliers
        inliers = np.where(distances < dis_threshold)[0]
        num_inliers = len(inliers)

        # Atualiza o melhor modelo se tiver mais inliers
        if num_inliers > best_num_inliers:
            best_inliers = inliers
            best_num_inliers = num_inliers

            # Atualiza N dinamicamente baseado no número de inliers
            w = num_inliers / num_points
            p_no_outliers = 1 - w**4
            N = int(np.log(1 - Ninl) / np.log(p_no_outliers)) + 1

    # Estima a homografia final H usando todos os inliers selecionados
    pts1_in = pts1[best_inliers, :]
    pts2_in = pts2[best_inliers, :]
    final_H = compute_normalized_dlt(pts1_in[:,0], pts2_in[:,0])

    return final_H, pts1_in, pts2_in


########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT


MIN_MATCH_COUNT = 10
img1 = cv.imread('monalisa01_1.jpg', 0)   # queryImage
img2 = cv.imread('monalisa01_2.jpg', 0)   # trainImage

# Inicialização do SIFT
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

    #################################################
    M, pts_1_in, pts_2_in = RANSAC(src_pts, dst_pts, 5, 100, 0.99) # 0.99 foi sugerido pela professora.
    #################################################

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

else:
    print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   flags=2)

img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 15))
plt.title("Estimativa de Homografia")

ax1 = plt.subplot(2, 2, 1)
plt.imshow(img3, 'gray')

ax2 = plt.subplot(2, 2, 2)
ax2.set_title('Primeira imagem')
plt.imshow(img1, 'gray')

ax3 = plt.subplot(2, 2, 3)
ax3.set_title('Segunda imagem')
plt.imshow(img2, 'gray')

ax4 = plt.subplot(2, 2, 4)
ax4.set_title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')

plt.show()
