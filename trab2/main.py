# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: 

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv


########################################################################################################################
# Função para normalizar pontos
# Entrada: points (pontos da imagem a serem normalizados)
# Saída: norm_points (pontos normalizados)
#        T (matriz de normalização)
def normalize_points(points):
    
    
    
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

# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)
def compute_normalized_dlt(pts1, pts2):
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
    H_matrix = vt[-1].reshape((3,3))
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
    
    # Define outros parâmetros como número de amostras do modelo, probabilidades da equação de N, etc 
    

    # Processo Iterativo
        # Enquanto não atende a critério de parada
        
        # Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2 
        
        # Usa as amostras para estimar uma homografia usando o DTL Normalizado

        # Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
        # o número de supostos inliers obtidos com o modelo estimado

        # Se o número de inliers é o maior obtido até o momento, guarda esse conjunto além das "s" amostras utilizadas. 
        # Atualiza também o número N de iterações necessárias

    # Terminado o processo iterativo
    # Estima a homografia final H usando todos os inliers selecionados.

    return H, pts1_in, pts2_in


########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT


MIN_MATCH_COUNT = 10
img1 = cv.imread('box.jpg', 0)   # queryImage
img2 = cv.imread('photo01a.jpg', 0)        # trainImage

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
    M = # AQUI ENTRA A SUA FUNÇÃO DE HOMOGRAFIA!!!!
    #################################################

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################
