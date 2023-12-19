import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
import sys

def calcular_medias(array):
    array = array[0]
    if array.shape != (1, 4, 2):
        raise ValueError("O array deve ter a forma (1, 4, 2).")

    # Calcula cx, a média dos 4 primeiros elementos
    cx = np.mean(array[0, :4], axis=0)

    # Calcula cy, a média dos 4 últimos elementos
    cy = np.mean(array[0, -4:], axis=0)

    return cx, cy


files_names = ["camera-00.mp4","camera-01.mp4","camera-02.mp4","camera-03.mp4"] 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters()

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
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None:
            corners_with_id_0 = [corners[i] for i in range(len(ids)) if ids[i] == 0]

        if len(corners_with_id_0) != 0:
            cX,cY =calcular_medias(corners_with_id_0)
            array.append((cX,cY))
        else:
            array.append(np.array([[]]))
        # frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)

        # cv2.imshow('output', frame_markers)
        
        
        
        if cv2.waitKey(1) == ord('q'):
            break
# ~m[ui,yi,1]
cam_0 = arrays[0]
cam_1 = arrays[1]
cam_2 = arrays[2]
cam_3 = arrays[3]
print(len(cam_3))


import json

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

projection_matrix = np.eye(3,4)



P=[[],[],[],[]]
eye_matrix = np.eye(3,4)

fsx0 = res0[0]*1
fsy0 = res0[1]*1
ox0 = res0[0]/2
oy0 = res0[1]/2
# K0 = np.array([[fsx0, 0,ox0], [0,fsy0, oy0],[0,0,1]])


# Matriz de rotação da câmera para o mundo (transposta)
R_camera_to_world = R0.T
t_camera_to_world = -R0.T @ T0
RT_camera_to_world = np.hstack((R_camera_to_world, t_camera_to_world))
projection_matrix = np.vstack((RT_camera_to_world, [0, 0, 0, 1]))

P_temp = np.dot(eye_matrix,projection_matrix)

P_temp = np.dot(K0,P_temp)

P[0]=P_temp


fsx1 = res1[0]*1
fsy1 = res1[1]*1
ox1 = res1[0]/2
oy1 = res1[1]/2
# K1 = np.array([[fsx1, 0,ox1], [0,fsy1, oy1],[0,0,1]])

# Matriz de rotação da câmera para o mundo (transposta)
R_camera_to_world = R1.T
t_camera_to_world = -R1.T @ T1
RT_camera_to_world = np.hstack((R_camera_to_world, t_camera_to_world))
projection_matrix = np.vstack((RT_camera_to_world, [0, 0, 0, 1]))

P_temp = np.dot(eye_matrix,projection_matrix)

P_temp = np.dot(K1,P_temp)

P[1]=P_temp

fsx2 = res2[0]*1
fsy2 = res2[1]*1
ox2 = res2[0]/2
oy2 = res2[1]/2
# K2 = np.array([[fsx2, 0,ox2], [0,fsy2, oy2],[0,0,1]])

# Matriz de rotação da câmera para o mundo (transposta)
R_camera_to_world = R2.T
t_camera_to_world = -R2.T @ T2
RT_camera_to_world = np.hstack((R_camera_to_world, t_camera_to_world))
projection_matrix = np.vstack((RT_camera_to_world, [0, 0, 0, 1]))

P_temp = np.dot(eye_matrix,projection_matrix)

P_temp = np.dot(K2,P_temp)

P[2]=P_temp


fsx3 = res3[0]*1
fsy3 = res3[1]*1
ox3 = res3[0]/2
oy3 = res3[1]/2
# K3 = np.array([[fsx3, 0,ox3], [0,fsy3, oy3],[0,0,1]])


# Matriz de rotação da câmera para o mundo (transposta)
R_camera_to_world = R3.T
t_camera_to_world = -R3.T @ T3
RT_camera_to_world = np.hstack((R_camera_to_world, t_camera_to_world))
projection_matrix = np.vstack((RT_camera_to_world, [0, 0, 0, 1]))

P_temp = np.dot(eye_matrix,projection_matrix)

P_temp = np.dot(K3,P_temp)

P[3]=P_temp

def verificar_e_concatenar_matrizes(matrizes):
    indices= []
    for indice, matriz in enumerate(matrizes):
       if matriz.size > 0:
            indices.append(indice)
    matrizes_nao_vazias = [m for m in matrizes if not np.all(m == 0)]

    if not matrizes_nao_vazias:
        return None  # Retorna None se todas as matrizes estiverem vazias
    
    
    if len(matrizes_nao_vazias) < 2:
        return print("nao rola")

    if len(matrizes_nao_vazias)==2:
        a = -matrizes_nao_vazias[0].reshape(-1, 1)
        a = np.vstack((a,-np.ones((1, 1))))
        b = -matrizes_nao_vazias[1].reshape(-1, 1)
        b = np.vstack((b,-np.ones((1, 1))))

        linha1 = np.hstack((P[indices[0]],a))
        linha1 = np.hstack((linha1,np.zeros((linha1.shape[0], 1))))
        
        linha2 = np.hstack((P[indices[1]],np.zeros((linha1.shape[0], 1))))
        linha2 = np.hstack((linha2,b))

        matriz_g = np.vstack((linha1,linha2))
        print("aaa")

    if len(matrizes_nao_vazias)==3:
        a = -matrizes_nao_vazias[0].reshape(-1, 1)
        a = np.vstack((a,-np.ones((1, 1))))
        b = -matrizes_nao_vazias[1].reshape(-1, 1)
        b = np.vstack((b,-np.ones((1, 1))))
        c = -matrizes_nao_vazias[2].reshape(-1, 1)
        c = np.vstack((c,-np.ones((1, 1))))

        linha1 = np.hstack((P[indices[0]],a))
        linha1 = np.hstack((linha1,np.zeros((linha1.shape[0], 1))))
        linha1 = np.hstack((linha1,np.zeros((linha1.shape[0], 1))))

        linha2 = np.hstack((P[indices[1]],np.zeros((linha1.shape[0], 1))))
        linha2 = np.hstack((linha2,b))
        linha2 = np.hstack((linha2,np.zeros((linha2.shape[0], 1))))

        linha3 = np.hstack((P[indices[2]],np.zeros((linha1.shape[0], 1))))
        linha3 = np.hstack((linha3,np.zeros((linha3.shape[0], 1))))
        linha3 = np.hstack((linha3,c))

        print("bbb")

        matriz_g = np.vstack((linha1,linha2,linha3))


    if len(matrizes_nao_vazias)==4:
        a = -matrizes_nao_vazias[0].reshape(-1, 1)
        a = np.vstack((a,-np.ones((1, 1))))
        b = -matrizes_nao_vazias[1].reshape(-1, 1)
        b = np.vstack((b,-np.ones((1, 1))))
        c = -matrizes_nao_vazias[2].reshape(-1, 1)
        c = np.vstack((c,-np.ones((1, 1))))
        d = -matrizes_nao_vazias[3].reshape(-1, 1)
        d = np.vstack((d,-np.ones((1, 1))))
        linha1 = np.hstack((P[indices[0]],a))
        linha1 = np.hstack((linha1,np.zeros((linha1.shape[0], 1))))
        linha1 = np.hstack((linha1,np.zeros((linha1.shape[0], 1))))
        linha1 = np.hstack((linha1,np.zeros((linha1.shape[0], 1))))

        linha2 = np.hstack((P[indices[1]],np.zeros((linha1.shape[0], 1))))
        linha2 = np.hstack((linha2,b))
        linha2 = np.hstack((linha2,np.zeros((linha2.shape[0], 1))))
        linha2 = np.hstack((linha2,np.zeros((linha2.shape[0], 1))))

        linha3 = np.hstack((P[indices[2]],np.zeros((linha1.shape[0], 1))))
        linha3 = np.hstack((linha3,np.zeros((linha3.shape[0], 1))))
        linha3 = np.hstack((linha3,c))
        linha3 = np.hstack((linha3,np.zeros((linha3.shape[0], 1))))

        linha4 = np.hstack((P[indices[3]],np.zeros((linha1.shape[0], 1))))
        linha4 = np.hstack((linha4,np.zeros((linha4.shape[0], 1))))
        linha4 = np.hstack((linha4,np.zeros((linha4.shape[0], 1))))
        linha4 = np.hstack((linha4,d))

        matriz_g = np.vstack((linha1,linha2,linha3,linha4))

    return matriz_g

total = len(cam_0)
i=0
print([np.array(cam_0[i][0]), cam_1[i][0], cam_2[i][0], cam_3[i][0]])
matrizes_M = [] 
while i < total :
    print(i)

    # Criar uma nova lista contendo as coordenadas (x, y, 1) para cada tupla
    matrizes = [cam_0[i][0], cam_1[i][0], cam_2[i][0], cam_3[i][0]] #aqui coloca as cordenadas do aruco dentro de um for
    # nova_lista = [np.hstack((tupla[0], 1)) for tupla in matrizes]

    # Converter a lista em uma matriz numpy
    # matrizes = np.array(nova_lista)
    
    matriz_concatenada = verificar_e_concatenar_matrizes(matrizes)
    matrizes_M.append(matriz_concatenada)
    i += 1
print(len(matrizes_M))
# matriz_concatenada = verificar_e_concatenar_matrizes(matrizes)
array_Vt = []
for matriz in matrizes_M:
    U, D, Vt = np.linalg.svd(matriz)
    print(Vt[-1, :4])
    array_Vt.append(Vt[-1, :4])

cv2.destroyAllWindows()

# Criar o gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Loop sobre os vetores Vt e plotar os pontos 3D
for Vt in array_Vt:
    x, y, z = Vt[0]/Vt[3], Vt[1]/Vt[3], Vt[2]/Vt[3]
    ax.scatter(x, y, z, c='r', marker='o')

# Definir os rótulos dos eixos
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')
ax.axis('equal')

# Mostrar o gráfico
plt.show()