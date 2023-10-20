import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QPushButton,QGroupBox
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
from stl import mesh
import numpy as np


###### Crie suas funções de translação, rotação, criação de referenciais, plotagem de setas e qualquer outra função que precisar

def get_obj_stl(obj): # recebe o arquivo STL para a elaboração do vetor obj
    your_mesh = mesh.Mesh.from_file(obj)
    x = your_mesh.x.flatten()
    y = your_mesh.y.flatten()
    z = your_mesh.z.flatten()
    return np.array([x.T,y.T,z.T,np.ones(x.size)])

def set_plot(ax = None, figure = None, lim = [-2, 2]):
    if figure == None:
        figure = plt.figure(figsize=(8,8))
    if ax == None:
        ax = plt.axes(projection='3d')
    #ax.set_title('camera reference')
    ax.set_xlim(lim)
    ax.set_xlabel('x axis')
    ax.set_ylim(lim)
    ax.set_ylabel('y axis')
    ax.set_zlim(lim)
    ax.set_zlabel('z axis')
    return ax

def draw_arrows(point,base,axis,length=5):
    # The object base is a matrix, where each column represents the vector
    # of one of the axis, written in homogeneous coordinates (ax,ay,az,0)
    # Plot vector of x-axis
    axis.quiver(point[0],point[1],point[2],base[0,0],base[1,0],base[2,0],color='red',pivot='tail',  length=length)
    # Plot vector of y-axis
    axis.quiver(point[0],point[1],point[2],base[0,1],base[1,1],base[2,1],color='green',pivot='tail',  length=length)
    # Plot vector of z-axis
    axis.quiver(point[0],point[1],point[2],base[0,2],base[1,2],base[2,2],color='blue',pivot='tail',  length=length)
    return axis

def world_translation(x, y, z): 
    translate_matrix = np.eye(4)
    translate_matrix[0, -1] = x
    translate_matrix[1, -1] = y
    translate_matrix[2, -1] = z
    return translate_matrix                    

def world_rotation(eixo, theta): # eixo = [x, y, z] ; theta em graus.
    theta = theta*np.pi/180 #
    if   eixo == 'x':
      rotation_matrix=np.array([[1,0,0,0],[0, np.cos(theta),-np.sin(theta),0],[0, np.sin(theta), np.cos(theta),0],[0,0,0,1]])
    elif eixo == 'y':
      rotation_matrix=np.array([[np.cos(theta),0, np.sin(theta),0],[0,1,0,0],[-np.sin(theta), 0, np.cos(theta),0],[0,0,0,1]])
    elif eixo == 'z':
      rotation_matrix=np.array([[np.cos(theta),-np.sin(theta),0,0],[np.sin(theta),np.cos(theta),0,0],[0,0,1,0],[0,0,0,1]])
    else:
      print('Eixo inexistente ou incorreto.')
    return rotation_matrix

def cam_translation(M_cam, x, y, z):
    M_inv = np.linalg.inv(M_cam)
    T = world_translation(x, y, z)
    translate_matrix = M_cam @ T @ M_inv
    return translate_matrix

def cam_rotation(M_cam, eixo, theta):
    M_inv = np.linalg.inv(M_cam)
    R = world_rotation(eixo, theta)
    rotation_matrix = M_cam @ R @ M_inv
    return rotation_matrix

def change_cam2world (M, point_cam):
      #Convert from camera frame to world frame
      p_world = np.dot(M, point_cam)
      return p_world

def change_world2cam (M, point_world):
      #Convert from world frame to camera frame
      M_inv = np.linalg.inv(M)
      p_cam = np.dot(M_inv, point_world)
      return p_cam

def house_example():
    house = np.array([[0,         0,         0],
            [0,  -10.0000,         0],
            [0, -10.0000,   12.0000],
            [0,  -10.4000,   11.5000],
            [0,   -5.0000,   16.0000],
            [0,         0,   12.0000],
            [0,    0.5000,   11.4000],
            [0,         0,   12.0000],
            [0,         0,         0],
    [-12.0000,         0,         0],
    [-12.0000,   -5.0000,         0],
    [-12.0000,  -10.0000,         0],
            [0,  -10.0000,         0],
            [0,  -10.0000,   12.0000],
    [-12.0000,  -10.0000,   12.0000],
    [-12.0000,         0,   12.0000],
            [0,         0,   12.0000],
            [0,  -10.0000,   12.0000],
            [0,  -10.5000,   11.4000],
    [-12.0000,  -10.5000,   11.4000],
    [-12.0000,  -10.0000,   12.0000],
    [-12.0000,   -5.0000,   16.0000],
            [0,   -5.0000,   16.0000],
            [0,    0.5000,   11.4000],
    [-12.0000,    0.5000,   11.4000],
    [-12.0000,         0,   12.0000],
    [-12.0000,   -5.0000,   16.0000],
    [-12.0000,  -10.0000,   12.0000],
    [-12.0000,  -10.0000,         0],
    [-12.0000,   -5.0000,         0],
    [-12.0000,         0,         0],
    [-12.0000,         0,   12.0000],
    [-12.0000,         0,         0]])

    house = np.transpose(house)

    #add a vector of ones to the house matrix to represent the house in homogeneous coordinates
    house = np.vstack([house, np.ones(np.size(house,1))])
    return house

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #definindo as variaveis
        self.set_variables()
        #Ajustando a tela    
        self.setWindowTitle("Grid Layout")
        self.setGeometry(100, 100,1280 , 720)
        self.setup_ui()

    def set_variables(self):
        self.objeto = get_obj_stl('donkey_kong.STL')
        # Adequação do objeto para o mundo.
        R = world_rotation('z', 90)
        T = world_translation(0, 0, -30)
        self.objeto = T @ R @ self.objeto

        # base vector values
        e1 = np.array([[1],[0],[0],[0]]) # X
        e2 = np.array([[0],[1],[0],[0]]) # Y
        e3 = np.array([[0],[0],[1],[0]]) # Z
        base = np.hstack((e1,e2,e3))
        #origin point
        point =np.array([[0],[0],[0],[1]])
        self.camera = np.hstack((base,point))
        self.referencial = self.camera

        self.px_base = 1280  
        self.px_altura = 720 
        self.dist_foc = 50 
        self.stheta = 0 
        self.ox = self.px_base/2 
        self.oy = self.px_altura/2 
        self.ccd = [36,24]
        self.projection_matrix = np.eye(3, 4)

    def setup_ui(self):
        # Criar o layout de grade
        grid_layout = QGridLayout()

        # Criar os widgets
        line_edit_widget1 = self.create_world_widget("Ref mundo")
        line_edit_widget2  = self.create_cam_widget("Ref camera")
        line_edit_widget3  = self.create_intrinsic_widget("Params instr.")

        self.canvas = self.create_matplotlib_canvas()

        # Adicionar os widgets ao layout de grade
        grid_layout.addWidget(line_edit_widget1, 0, 0)
        grid_layout.addWidget(line_edit_widget2, 0, 1)
        grid_layout.addWidget(line_edit_widget3, 0, 2)
        grid_layout.addWidget(self.canvas, 1, 0, 1, 3)

        # Criar um widget para agrupar o botão de reset
        reset_widget = QWidget()
        reset_layout = QHBoxLayout()
        reset_widget.setLayout(reset_layout)

        # Criar o botão de reset vermelho
        reset_button = QPushButton("Reset")
        reset_button.setFixedSize(50, 30)  # Define um tamanho fixo para o botão (largura: 50 pixels, altura: 30 pixels)
        style_sheet = """
            QPushButton {
                color : white ;
                background: rgba(255, 127, 130,128);
                font: inherit;
                border-radius: 5px;
                line-height: 1;
            }
        """
        reset_button.setStyleSheet(style_sheet)
        reset_button.clicked.connect(self.reset_canvas)

        # Adicionar o botão de reset ao layout
        reset_layout.addWidget(reset_button)

        # Adicionar o widget de reset ao layout de grade
        grid_layout.addWidget(reset_widget, 2, 0, 1, 3)

        # Criar um widget central e definir o layout de grade como seu layout
        central_widget = QWidget()
        central_widget.setLayout(grid_layout)
        
        # Definir o widget central na janela principal
        self.setCentralWidget(central_widget)

    def create_intrinsic_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['n_pixels_base:', 'n_pixels_altura:', 'ccd_x:', 'ccd_y:', 'dist_focal:', 'sθ:']  # Texto a ser exibido antes de cada QLineEdit

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()  # Validador numérico
            line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        ##### Você deverá criar, no espaço reservado ao final, a função self.update_params_intrinsc ou outra que você queira 
        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_params_intrinsc(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        # Retornar o widget e a lista de caixas de texto
        return line_edit_widget
    
    def create_world_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']  # Texto a ser exibido antes de cada QLineEdit

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()  # Validador numérico
            line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        ##### Você deverá criar, no espaço reservado ao final, a função self.update_world ou outra que você queira 
        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_world(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        # Retornar o widget e a lista de caixas de texto
        return line_edit_widget

    def create_cam_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']  # Texto a ser exibido antes de cada QLineEdit

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()  # Validador numérico
            line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        ##### Você deverá criar, no espaço reservado ao final, a função self.update_cam ou outra que você queira 
        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_cam(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        # Retornar o widget e a lista de caixas de texto
        return line_edit_widget

    def create_matplotlib_canvas(self):
        # Criar um widget para exibir os gráficos do Matplotlib
        canvas_widget = QWidget()
        canvas_layout = QHBoxLayout()
        canvas_widget.setLayout(canvas_layout)

        # Criar um objeto FigureCanvas para exibir o gráfico 2D
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_title("Imagem")
        self.canvas1 = FigureCanvas(self.fig1)

        ##### Falta acertar os limites do eixo X
        self.ax1.set_xlim([0, self.px_base])
        ##### Falta acertar os limites do eixo Y
        self.ax1.set_ylim([self.px_altura, 0])
        ##### Você deverá criar a função de projeção 
        obj_2d = self.projection_2d()

        ##### Falta plotar o object_2d que retornou da projeção
        self.ax1.plot(obj_2d[0, :], obj_2d[1, :])
        self.ax1.grid('True')
        self.ax1.set_aspect('equal')  
        canvas_layout.addWidget(self.canvas1)

        # Criar um objeto FigureCanvas para exibir o gráfico 3D
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        self.ax2 = set_plot(ax=self.ax2, lim = [-40, 40])
        
        # Objeto
        self.ax2.plot(self.objeto[0, :], self.objeto[1, :], self.objeto[2, :], 'b' )
        # Camera
        draw_arrows(self.camera[:,-1], self.camera[:,0:3], self.ax2)
        # Referencial no mundo
        draw_arrows(self.referencial[:,-1], self.referencial[:,0:3], self.ax2)

        self.canvas2 = FigureCanvas(self.fig2)
        canvas_layout.addWidget(self.canvas2)

        # Retornar o widget de canvas
        return canvas_widget


    ##### Você deverá criar as suas funções aqui
    
    def update_params_intrinsc(self, line_edits):
        data = [self.px_base, self.px_altura, self.ccd[0], self.ccd[1], self.dist_foc, self.stheta]
        for i in range(len(line_edits)):
            try:
                value = float(line_edits[i].text())
                data[i] = value
            except: None   
        self.px_base    = float(data[0])
        self.px_altura  = float(data[1])
        self.ccd[0]     = float(data[2])
        self.ccd[1]     = float(data[3])
        self.dist_foc   = float(data[4])
        self.stheta     = float(data[5])
        self.update_canvas()
        [i.clear() for i in line_edits]
        return 

    def update_world(self,line_edits):
        data = []
        for i in line_edits:
            try:
                value = float(i.text())
                data.append(value)
            except:
                data.append(float(0))  
        x_move  = float(data[0])
        y_move  = float(data[2])
        z_move  = float(data[4])
        x_angle = float(data[1])
        y_angle = float(data[3])
        z_angle = float(data[5])
        
        T = world_translation(x_move, y_move, z_move)

        Rx = world_rotation('x', x_angle)
        Ry = world_rotation('y', y_angle)
        Rz = world_rotation('z', z_angle)
        R = Rx @ Ry @ Rz
        
        M = T @ R
        self.camera = np.dot(M, self.camera)
        self.update_canvas()
        [i.clear() for i in line_edits]
        return

    def update_cam(self,line_edits):
        data = []
        for i in line_edits:
            try:
                value = float(i.text())
                data.append(value)
            except:
                data.append(float(0))  
        x_move  = float(data[0])
        y_move  = float(data[2])
        z_move  = float(data[4])
        x_angle = float(data[1])
        y_angle = float(data[3])
        z_angle = float(data[5])

        T = cam_translation(self.camera, x_move, y_move, z_move)

        Rx = cam_rotation(self.camera, 'x', x_angle)
        Ry = cam_rotation(self.camera, 'y', y_angle)
        Rz = cam_rotation(self.camera, 'z', z_angle)
        R = Rx @ Ry @ Rz
        
        M = T @ R   # Foi escolhida esta ordem de acontecimentos.
        self.camera = np.dot(M, self.camera)
        self.update_canvas()
        [i.clear() for i in line_edits]
        return 
    
    def projection_2d(self):
        Cam_inv = np.linalg.inv(self.camera)
        MPI = self.generate_intrinsic_params_matrix()
        obj_2d = MPI @ self.projection_matrix @ Cam_inv @ self.objeto
        obj_2d[0, :] = obj_2d[0, :] / obj_2d[2, :]
        obj_2d[1, :] = obj_2d[1, :] / obj_2d[2, :]
        obj_2d[2, :] = obj_2d[2, :] / obj_2d[2, :]
        return obj_2d
    
    def generate_intrinsic_params_matrix(self):
        fs_x = self.px_base * self.dist_foc / self.ccd[0]
        fs_y = self.px_altura * self.dist_foc / self.ccd[1]
        fs_theta = self.stheta * self.dist_foc
        ox = self.px_base / 2
        oy = self.px_altura / 2
        MPI = array([[fs_x, fs_theta, ox],
                     [   0,     fs_y, oy],
                     [   0,        0,  1]])
        return MPI
    
    def update_canvas(self):
        plt.close('all')
        
        # Parte 2D
        obj_2d = self.projection_2d()
        self.ax1.clear()
        self.ax1.set_xlim([0, self.px_base])
        self.ax1.set_ylim([self.px_altura, 0])
        self.ax1.plot(obj_2d[0, :], obj_2d[1, :])
        self.ax1.grid(True)
        self.ax1.set_aspect('equal')

        # Parte 3D
        self.ax2.clear()
        self.ax2 = set_plot(ax=self.ax2, lim=[-40, 40])
        self.ax2.plot3D(self.objeto[0, :], self.objeto[1, :], self.objeto[2, :], 'b')
        draw_arrows(self.camera[:,-1], self.camera[:,0:3], self.ax2)
        draw_arrows(self.referencial[:,-1], self.referencial[:,0:3], self.ax2)

        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas.layout().itemAt(1).widget().draw()
        return 
    
    def reset_canvas(self):
        self.set_variables()
        self.update_canvas()
        return
    
    def posicionar_cam(self):
        # Posicionamento da câmera fora da origem
        T = world_translation(40, 0, 0) 
        self.camera = T @ self.referencial
        R = cam_rotation(self.camera, 'y', -90)
        self.camera = R @ self.camera
        R = cam_rotation(self.camera, 'z', 90)
        self.camera = R @ self.camera
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
