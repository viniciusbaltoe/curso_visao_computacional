# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-= Imports =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
import cv2
import json
import numpy as np
#from cv2 import aruco
import matplotlib.pyplot as plt
import sys


# -=-=-=-=-=-=-=-=-=-= Retirado do arquivo parametros.py =-=-=-=-=-=-=-=-=-=-=-

# Function to read the intrinsic and extrinsic parameters of each camera
def camera_parameters(file):
    camera_data = json.load(open(file))
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'],
           camera_data['resolution']['height']]
    tf = np.array(camera_data['extrinsic']['tf']['doubles']).reshape(4, 4)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis


#Load cameras parameters
K0, R0, T0, res0, dis0 = camera_parameters('0.json')
K1, R1, T1, res1, dis1 = camera_parameters('1.json')
K2, R2, T2, res2, dis2 = camera_parameters('2.json')
K3, R3, T3, res3, dis3 = camera_parameters('3.json')

# -=-=-=-=-=-= Transformação que converte do mundo para a câmera =-=-=-=-=-=-=-=-

def calcular_P(R, T, K, eye_matrix):
    R_camera_to_world = R.T
    t_camera_to_world = -R.T @ T
    RT_camera_to_world = np.hstack((R_camera_to_world, t_camera_to_world))
    projection_matrix = np.vstack((RT_camera_to_world, [0, 0, 0, 1]))

    P_temp = np.dot(eye_matrix, projection_matrix)
    P_temp = np.dot(K, P_temp)

    return P_temp

# Usar a função para calcular os P's
projection_matrix = np.eye(3, 4)
eye_matrix = np.eye(3, 4)

P = []

P.append(calcular_P(R0, T0, K0, eye_matrix))
P.append(calcular_P(R1, T1, K1, eye_matrix))
P.append(calcular_P(R2, T2, K2, eye_matrix))
P.append(calcular_P(R3, T3, K3, eye_matrix))

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def calcular_medias(array):
    array = array[0]
    if array.shape != (1, 4, 2):
        raise ValueError("O array deve ter a forma (1, 4, 2).")

    # Calcula cx, a média dos 4 primeiros elementos
    cx = np.mean(array[0, :4], axis=0)

    # Calcula cy, a média dos 4 últimos elementos
    cy = np.mean(array[0, -4:], axis=0)

    return cx, cy

def detectar_aruco(files_names):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    
    arrays = []
    while files_names:
        vid = cv2.VideoCapture(files_names[0])
        array=[]
        while True:
            _, img = vid.read()
            if img is None:
                files_names.pop(0)            
                arrays.append(array)
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            if ids is not None:
                corners_with_id_0 = [corners[i] for i in range(len(ids)) if ids[i] == 0]

            if len(corners_with_id_0) != 0:
                cX, cY = calcular_medias(corners_with_id_0)
                array.append((cX, cY))
            else:
                array.append(np.array([[]]))
                
            if cv2.waitKey(1) == ord('q'):
                break
                
    return arrays

def verificar_e_concatenar_matrizes(matrizes, P):
    indices = [indice for indice, matriz in enumerate(matrizes) if matriz.size > 0]
    matrizes_nao_vazias = [m for m in matrizes if not np.all(m == 0)]

    if not matrizes_nao_vazias or len(matrizes_nao_vazias) < 2:
        return None  # Retorna None se todas as matrizes estiverem vazias ou menos de 2 matrizes não vazias
    
    matriz_g = np.zeros((0, 0))

    for i, matriz_nao_vazia in enumerate(matrizes_nao_vazias):
        a = -matriz_nao_vazia.reshape(-1, 1)
        a = np.vstack((a, -np.ones((1, 1))))

        linha = np.hstack((P[indices[i]], a))
        linha = np.hstack((linha, np.zeros((linha.shape[0], 1))))

        matriz_g = np.vstack((matriz_g, linha))

    return matriz_g

def plotar_grafico_3D(array_Vt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for Vt in array_Vt:
        x, y, z = Vt[0] / Vt[3], Vt[1] / Vt[3], Vt[2] / Vt[3]
        ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.set_zlabel('Eixo Z')
    ax.axis('equal')

    plt.show()

# Uso das funções
files_names = ["camera-00.mp4", "camera-01.mp4", "camera-02.mp4", "camera-03.mp4"]
arrays = detectar_aruco(files_names)

cam_0, cam_1, cam_2, cam_3 = arrays[0], arrays[1], arrays[2], arrays[3]

total = len(cam_0)
i = 0
matrizes_M = []

while i < total:
    print(i)
    matrizes = [cam_0[i][0], cam_1[i][0], cam_2[i][0], cam_3[i][0]]
    matriz_concatenada = verificar_e_concatenar_matrizes(matrizes, P)
    matrizes_M.append(matriz_concatenada)
    i += 1

#print(len(matrizes_M))
array_Vt = []

for matriz in matrizes_M:
    U, D, Vt = np.linalg.svd(matriz)
    print(Vt[-1, :4])
    array_Vt.append(Vt[-1, :4])

cv2.destroyAllWindows()
plotar_grafico_3D(array_Vt)