import numpy as np
from sklearn.svm import SVR
from PLOT import plot_grid_S
from read_grid import Coord , wallNodes , Grid

xCoord = np.array([Coord[:, 0]]).T
yCoord = np.array([Coord[:, 1]]).T
nNodes = Coord.shape[0]
nWallNodes = len(wallNodes)

beta = 1       #SVR参数
dismin = float('inf')
s = 1

xCoord_new = np.array([xCoord[:, 0]]).T #避免改变原数组
yCoord_new = np.array([yCoord[:, 0]]).T

# 对物面点进行变形
dy = np.zeros((nWallNodes, 1))
for i in range(nWallNodes):
    wall_index = wallNodes[i] - 1
    dy[i,0] = 0.2 * np.sin(-2*np.pi * xCoord_new[wall_index][0] )
    yCoord_new[wall_index] += dy[i,0]

for i in range(Grid.shape[0]):
        dis = np.sqrt((xCoord_new[Grid[i][0] - 1] - xCoord_new[Grid[i][1] - 1]) ** 2 + (yCoord_new[Grid[i][0] - 1] - yCoord_new[Grid[i][1] - 1]) ** 2) + 1e-40
        if dis < dismin:
            dismin = dis

# clf = SVR(kernel='rbf', C=pow(10,6) , gamma=0.125 , epsilon=0.1 )
clf = SVR(kernel='rbf', C=pow(10,6) , gamma=0.125 , epsilon=beta * dismin[0] / s)
x_train = Coord[wallNodes-1]
y_train = dy.ravel()
clf.fit(x_train , y_train)
dy_pred = clf.predict(Coord)

yCoord_pred = yCoord.copy()
for i in range(nNodes):
    yCoord_pred[i] = yCoord[i] + dy_pred[i]

# 绘制网格
plot_grid_S(Grid, xCoord_new, yCoord_pred)
