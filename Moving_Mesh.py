import numpy as np
from sklearn.svm import SVR
from PLOT import plot_grid
from read_grid import Coord , wallNodes , Grid

xCoord = np.array([Coord[:, 0]]).T
yCoord = np.array([Coord[:, 1]]).T
nNodes = Coord.shape[0]
nWallNodes = len(wallNodes)

#参数
lamda = 1      #波长
c = 0.1        #波速
v = -0.2       #游动速度
T = 2.0        #周期
t = 0          #起始时间
dt = 0.5       #时间间隔

L = 1.0        #长度
beta = 1       #回归阈值系数
#网格点间最小距离
dismin = float("inf") 
s = 0          #变形迭代次数

while t < 10:
    dy = np.zeros((nWallNodes,1))
    xCoord_new = np.array([xCoord[:, 0]]).T #避免改变原数组
    yCoord_new = np.array([yCoord[:, 0]]).T
    t = t + dt
    s += 1

    # 对物面点进行变形
    for i in range(nWallNodes):
        wall_index = wallNodes[i] - 1
        xCoord_new[wall_index] = xCoord[wall_index] + v * t
    nose_x = min(xCoord_new[wallNodes-1])

    for i in range(nWallNodes):
        wall_index = wallNodes[i] - 1
        x = xCoord_new[wall_index] - nose_x
        y = yCoord[wall_index]

        #此处表示已知所有物面点的位移
        A = min(1, t / T) * (0.02 - 0.0825 * x + 0.1625 * x**2)
        B = A * np.sin(2 * np.pi / lamda * (x - c * t))
        dy[i,0] = B[0]
        yCoord_new[wall_index] = yCoord[wall_index] + B[0]

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

    for i in range(nNodes):
        if i in wallNodes - 1:
            continue
        dx = v * t  
        xCoord_new[i] = xCoord_new[i] + dx 
    # # 计算权重系数矩阵W
    # fai = np.zeros((nWallNodes, nWallNodes))

    # for i in range(nWallNodes):
    #     wall_index = wallNodes[i] - 1
    #     x1 = xCoord[wall_index]
    #     y1 = yCoord[wall_index]
    #     for j in range(nWallNodes):
    #         wall_index2 = wallNodes[j] - 1
    #         x2 = xCoord[wall_index2]
    #         y2 = yCoord[wall_index2]
            
    #         #距离加上1e-40防止除以零
    #         dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + 1e-40
    #         fai[i, j] = RBF.RBF_func(dis[0], r0, basis)
    # W = np.dot(np.linalg.inv(fai),dy)

    # # 利用W计算内场点的位移
    # # 这里将远场边界点通过插值更改了坐标，不清楚是否需要保持不变或者保持整个框架不变
    # fai = np.zeros((1, nWallNodes))

    # for i in range(nNodes):
    #     xNode = xCoord[i]
    #     yNode = yCoord[i]
    #     if i in wallNodes - 1:
    #         continue
    #     for j in range(nWallNodes):
    #         wall_index = wallNodes[j] - 1
    #         xw = xCoord[wall_index]
    #         yw = yCoord[wall_index]

    #         dis = np.sqrt((xNode - xw)**2 + (yNode - yw)**2) + 1e-40
    #         fai[0, j] = RBF.RBF_func(dis[0], r0, basis)

    #     dy = np.dot(fai[0, :] , W)  
    #     yCoord_new[i] = yCoord_new[i] + dy[0]  

    # 绘制网格
    plot_grid(Grid, xCoord_new , yCoord_pred , nose_x)
    if input(f"t={t} 按任意键继续生成下一时刻网格，或输入q退出循环:\n") == 'q':
        break

