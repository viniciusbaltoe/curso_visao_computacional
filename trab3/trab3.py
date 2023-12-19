import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
import json

def calculate_means(array):
    array = array[0]
    if array.shape != (1, 4, 2):
        raise ValueError("Array in the wrong format.")
    cx = np.mean(array[0, :4], axis=0)
    cy = np.mean(array[0, -4:], axis=0)
    return cx, cy

def check_and_concatenate_matrices(matrices):
    indices = [index for index, matrix in enumerate(matrices) if matrix.size > 0]
    non_empty_matrices = [m for m in matrices if not np.all(m == 0)]
    if not non_empty_matrices or len(non_empty_matrices) < 2:
        return None
    lines = []
    for i, matrix in enumerate(non_empty_matrices):
        a = -matrix.reshape(-1, 1)
        a = np.vstack((a, -np.ones((1, 1))))
        line = np.hstack((P[indices[i]], a))
        for j, _ in enumerate(non_empty_matrices):
            if j != i:
                line = np.hstack((line, np.zeros((line.shape[0], 1))))
            else:
                line = np.hstack((line, -np.ones((line.shape[0], 1))))
        lines.append(line)
    matrix_g = np.vstack(lines)
    return matrix_g

# Load camera parameters
def camera_parameters(file):
    camera_data = json.load(open(file))
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'], camera_data['resolution']['height']]
    tf = np.array(camera_data['extrinsic']['tf']['doubles']).reshape(4, 4)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis

# Initialize ArUco detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
files_names = ["camera-00.mp4", "camera-01.mp4", "camera-02.mp4", "camera-03.mp4"] 

arrays = []

while files_names:
    vid = cv2.VideoCapture(files_names[0])
    array = []
    while True:
        _, img = vid.read()
        if img is None:
            files_names.pop(0)            
            arrays.append(array)
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ArUco detection
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)  # Alteração aqui
        if ids is not None:
            corners_with_id_0 = [corners[i] for i in range(len(ids)) if ids[i] == 0]
        if len(corners_with_id_0) != 0:
            cX, cY = calculate_means(corners_with_id_0)
            array.append((cX, cY))
        else:
            array.append(np.array([[]]))
        if cv2.waitKey(1) == ord('q'):
            break

cameras = [arrays[i] for i in range(len(arrays))]
print(len(cameras[-1]))

# Load cameras parameters
camera_params = [camera_parameters(f'{i}.json') for i in range(4)]

projection_matrix = np.eye(3, 4)
P = [[], [], [], []]
eye_matrix = np.eye(3, 4)

for i, params in enumerate(camera_params):
    fsx = params[3][0] * 1
    fsy = params[3][1] * 1
    ox = params[3][0] / 2
    oy = params[3][1] / 2

    R_camera_to_world = params[1].T
    t_camera_to_world = -params[1].T @ params[2]
    RT_camera_to_world = np.hstack((R_camera_to_world, t_camera_to_world))
    projection_matrix = np.vstack((RT_camera_to_world, [0, 0, 0, 1]))

    P_temp = np.dot(eye_matrix, projection_matrix)
    P_temp = np.dot(params[0], P_temp)
    P[i] = P_temp

total = len(cameras[0])
i = 0
matrices_M = [] 

while i < total:
    print(i)
    matrices = [cam[i][0] for cam in cameras]
    concatenated_matrix = check_and_concatenate_matrices(matrices)
    if concatenated_matrix is not None:
        matrices_M.append(concatenated_matrix)
    i += 1

array_Vt = []
for matrix in matrices_M:
    U, D, Vt = np.linalg.svd(matrix)
    print(Vt[-1, :4])
    array_Vt.append(Vt[-1, :4])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for Vt in array_Vt:
    x, y, z = Vt[0]/Vt[3], Vt[1]/Vt[3], Vt[2]/Vt[3]
    ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_box_aspect([1, 1, 1]) #depending on your needs


plt.show()
