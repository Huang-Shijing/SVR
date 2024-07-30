import numpy as np
from sklearn.svm import SVR
from PLOT import plot_grid_S
from read_grid import Coord , wallNodes , Grid

xCoord = np.array([Coord[:, 0]]).T
yCoord = np.array([Coord[:, 1]]).T
nNodes = Coord.shape[0]
nWallNodes = len(wallNodes)

beta = 1       #SVR参数
ang = np.pi * 15/180
TCoord = (-0.25 , 0)
dismin = float('inf')
s = 1

xCoord_new = np.array([xCoord[:, 0]]).T #避免改变原数组
yCoord_new = np.array([yCoord[:, 0]]).T

# 对物面点进行变形
dy = np.zeros((nWallNodes, 1))
dx = np.zeros((nWallNodes, 1))
for i in range(nWallNodes):
    wall_index = wallNodes[i] - 1
    yCoord_new[wall_index] = TCoord[1] + (xCoord[wall_index] - TCoord[0]) * np.sin(ang) + (yCoord[wall_index] - TCoord[1]) * np.cos(ang)
    xCoord_new[wall_index] = TCoord[0] + (xCoord[wall_index] - TCoord[0]) * np.cos(ang) - (yCoord[wall_index] - TCoord[1]) * np.sin(ang)
    dy[i,0] = yCoord_new[wall_index][0] - yCoord[wall_index][0]
    dx[i,0] = xCoord_new[wall_index][0] - xCoord[wall_index][0]

for i in range(Grid.shape[0]):
        dis = np.sqrt((xCoord_new[Grid[i][0] - 1] - xCoord_new[Grid[i][1] - 1]) ** 2 + (yCoord_new[Grid[i][0] - 1] - yCoord_new[Grid[i][1] - 1]) ** 2) + 1e-40
        if dis < dismin:
            dismin = dis

x_clf = SVR(kernel='rbf', C=pow(10,6) , gamma=0.125 , epsilon=beta * dismin[0] / s)
x_train = Coord[wallNodes-1]
y_train = dx.ravel()
x_clf.fit(x_train , y_train)
dx_pred = x_clf.predict(Coord)

y_clf = SVR(kernel='rbf', C=pow(10,6) , gamma=0.125 , epsilon=beta * dismin[0] / s)
x_train = Coord[wallNodes-1]
y_train = dy.ravel()
y_clf.fit(x_train , y_train)
dy_pred = y_clf.predict(Coord)

xCoord_pred = np.array([xCoord[:, 0]]).T
yCoord_pred = np.array([yCoord[:, 0]]).T
for i in range(nNodes):
    xCoord_pred[i] = xCoord[i] + dx_pred[i]
    yCoord_pred[i] = yCoord[i] + dy_pred[i]


# 绘制网格
plot_grid_S(Grid, xCoord_pred, yCoord_pred)
