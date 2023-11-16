# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: Artur Henrique do Nascimento Souza

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


########################################################################################################################


def normalize_points(points):
    # Calculate centroid
    centroid = np.mean(points, axis=0)

    # Calculate the average distance of the points from the centroid
    distances = np.linalg.norm(points - centroid, axis=1)
    avg_distance = np.mean(distances)

    # Define the scale factor to have the average distance as sqrt(2)
    scale_factor = np.sqrt(2) / avg_distance

    # Define the normalization matrix
    T = np.array([[scale_factor, 0, -scale_factor * centroid[0]],
                  [0, scale_factor, -scale_factor * centroid[1]],
                  [0, 0, 1]])

    # Apply normalization matrix to points
    homogenous_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    norm_points = np.dot(T, homogenous_points.T).T[:, :2]

    return norm_points, T

def my_homography(pts1, pts2):
    # Normaliza pontos
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)
    # Constrói o sistema de equações empilhando a matriz A de cada par de pontos correspondentes normalizados
    A = []
    for i in range(len(norm_pts1)):
        x, y = norm_pts1[i]
        u, v = norm_pts2[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A)

    # Calcula o SVD da matriz A_empilhada e estima a homografia H_normalizada
    try:
        U, D, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        # Tratamento da exceção
        print("SVD did not converge. Skipping iteration.")
        return []  # Pular a iteração
    U, D, Vt = np.linalg.svd(A)
    H_normalized = Vt[-1].reshape(3, 3)
    # Denormaliza H_normalizada e obtém H
    H = np.dot(np.dot(np.linalg.inv(T2), H_normalized), T1)
    return H


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


def RANSAC(pts1, pts2, dis_threshold, N, Ninl=0.99):
    
    # Define outros parâmetros como número de amostras do modelo, probabilidades da equação de N, etc 
    best_inliers = []
    best_num_inliers = 0
    iterations = 0
    s=4
    # Processo Iterativo
    while iterations < N:
      #  print(np.round(iterations*100/N),'%')
        # Enquanto não atende a critério de parada

        # Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2 

        pts_s_ind       = np.random.rand(s)*len(pts1)
        pts_s_ind       = pts_s_ind.astype(int)
        
        sampled_pts1    = pts1[pts_s_ind]
        sampled_pts2    = pts2[pts_s_ind]

        #print(sampled_pts1)
        # Usa as amostras para estimar uma homografia usando o DTL Normalizado
        H = my_homography(sampled_pts1[:,0],sampled_pts2[:,0])

        if not len(H):
            iterations += 1
            continue

        # Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
        # o número de supostos inliers obtidos com o modelo estimado
        inliers     = []
        num_inliers = 0
        lixo = []
        for i in range(len(pts1)):
            x1, y1 = pts1[i][0]
            x2, y2 = pts2[i][0]
            abc = [x1,y1,1]
            #print(abc)
            lixo.append(abc)
            transformed_pts = np.dot(H,abc)
            print(transformed_pts)
            transformed_pts /= transformed_pts[2]  # Normalizar pela coordenada homogênea
            print(transformed_pts)
            distance = np.sqrt((transformed_pts[0] - x2) ** 2 + (transformed_pts[1] - y2) ** 2)
            print(distance)
            #print(distance)
            if distance <= dis_threshold:
                inliers.append(i)
                num_inliers += 1
        #print(len(inliers))
        # Se o número de inliers é o maior obtido até o momento, guarda esse conjunto além das "s" amostras utilizadas. 
        if num_inliers > best_num_inliers:
            best_inliers = inliers
            best_num_inliers = num_inliers
        # Atualiza também o número N de iterações necessárias
        N = np.log(1 - Ninl) / np.log(1 - (best_num_inliers / len(pts1)) ** s)
        iterations += 1

    # Terminado o processo iterativo
    # Estima a homografia final H usando todos os inliers selecionados.
    pts1_in = pts1[best_inliers]
    pts2_in = pts2[best_inliers]
    H = my_homography(pts1_in[:,0], pts2_in[:,0])

    return H #, pts1_in, pts2_in


########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT


MIN_MATCH_COUNT = 10
img1 = cv.imread('monalisa01_1.jpg', 0)   # queryImage
img2 = cv.imread('monalisa01_2.jpg', 0)        # trainImage


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
    M = RANSAC(src_pts,dst_pts,5,20)# AQUI ENTRA A SUA FUNÇÃO DE HOMOGRAFIA!!!!
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
