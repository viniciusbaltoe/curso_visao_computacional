import sys
import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
import json

# Função para calcular os meios dos pontos em um array
def calculate_means(a):
    a = a[0]
    if a.shape != (1, 4, 2):
        raise ValueError("The array should have shape (1, 4, 2).")

    cx = np.mean(a[0, :4], axis=0)
    cy = np.mean(a[0, -4:], axis=0)
    return cx, cy

# Função para obter os parâmetros da câmera a partir de um arquivo JSON
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

# Função para concatenar e verificar matrizes
def concatenate_and_verify(matrix, P):
    non_empty_indices = [index for index, matrix in enumerate(matrix) if matrix[0].size > 0]
    non_empty_matrix = [m[0] for m in matrix if m[0].size > 0]

    if not non_empty_matrix or len(non_empty_matrix) < 2:
        return None

    len_matrix = len(non_empty_matrix)
    matrix_result = None

    if len_matrix > 1:
        lines = []
        for i in range(len_matrix):
            a = -non_empty_matrix[i].reshape(-1, 1)
            a = np.vstack((a, -np.ones((1, 1))))
            
            line = np.hstack((P[non_empty_indices[i]], np.zeros((a.shape[0], i)), a, np.zeros((a.shape[0], len_matrix - i - 1))))
            lines.append(line)

        matrix_result = np.vstack(lines)

    return matrix_result

# Configuração do dicionário e parâmetros do ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
file_list = ["camera-00.mp4", "camera-01.mp4", "camera-02.mp4", "camera-03.mp4"]

video_marker_positions = []
camera_projection_matrices = []
current_video_marker_positions = []

# Loop para processar cada vídeo
for idx, file in enumerate(file_list):
    vid = cv2.VideoCapture(file)
    current_frame_marker_positions = []

    window_name = f"Video {idx}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Loop para processar cada frame do vídeo
    while True:
        ret, img = vid.read()
        if not ret or img is None:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        corners_with_id_0 = [corners[i] for i in range(len(ids)) if ids[i] == 0] if ids is not None else []

        if corners_with_id_0:
            cX, cY = calculate_means(corners_with_id_0)
            current_frame_marker_positions.append((cX, cY))
        else:
            current_frame_marker_positions.append(np.array([[]]))

        cv2.imshow(window_name, img)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyWindow(window_name)
    video_marker_positions.append(current_frame_marker_positions)

    # Obter parâmetros da câmera e construir a matriz de projeção
    K, R, T, res, dis = camera_parameters(f"{idx}.json")
    fsx = res[0] * 1
    fsy = res[1] * 1
    ox = res[0] / 2
    oy = res[1] / 2

    Rcamtw = R.T
    tcamtw = -R.T @ T
    RT_camtw = np.hstack((Rcamtw, tcamtw))
    proj_m = np.concatenate((RT_camtw, [[0, 0, 0, 1]]), axis=0)

    P_t = np.eye(3, 4) @ proj_m
    P_t = K @ P_t
    camera_projection_matrices.append(P_t)

# Processamento para calcular e visualizar os pontos 3D
matrices = []
for i in range(len(video_marker_positions[0])):
    frame_matrices = [np.array(video_marker_positions[j][i]) for j in range(len(file_list))]
    concatenated_matrix = concatenate_and_verify(frame_matrices, camera_projection_matrices)
    matrices.append(concatenated_matrix)

resulting_A = []
for matrix in matrices:
    U, D, Vt = np.linalg.svd(matrix)
    resulting_A.append(Vt[-1, :4])

cv2.destroyAllWindows()

# Visualização dos pontos 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for Vt in resulting_A:
    x, y, z, w = Vt[0], Vt[1], Vt[2], Vt[3]
    ax.scatter(x/w, y/w, z/w, c='r', marker='o')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.show()
