'''

made by yang xing yeeeeee
1.取得偵測範圍內障礙物(x:-15~15, y:0~15)
2.初始化,在地圖上隨機灑點(最小單位1 int)
3.移動預測點(掉到地圖外的還沒處理)
4.權重更新(貝氏濾波)(取得被laser偵測倒的障礙物位置)
5.重新取樣
6.重複3~5直到結束

'''

# useful plt function:
#     plt.waitforbuttonpress()

# trouble 1:點不能灑在障礙上

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math


def init(x_range, y_range, hdg_range, N):
	particles = np.empty([N, 3]) # 二維, N個模擬點
	particles[:, 0] = np.random.randint(x_range[0], x_range[1], size=N) # 從頭到尾隨機產生範圍內x值
	particles[:, 1] = np.random.randint(y_range[0], y_range[1], size=N) # y值
	particles[:, 2] = np.random.uniform(hdg_range[0], hdg_range[1], size=N) # 角度
	particles[:, 2] %= 2 * np.pi # 轉成角度
	return particles

def predict(particles, input, noise, dt=1.0):
    N = len(particles)
    # 更新方向
    particles[:, 2] += input[0] + (np.random.randn(N) * noise[0])
    particles[:, 2] %= 2 * np.pi

    # 往車子本身前進方向移動(加上一些干擾參數)
    dist = (input[1] * dt) + (np.random.randn(N) * noise[1])
    particles[:, 0] += np.rint(np.cos(particles[:, 2]) * dist)
    particles[:, 1] += np.rint(np.sin(particles[:, 2]) * dist)

def update(particles, weights, dt_total, R, ob_detected):
    weights.fill(1.)
    i = 0
    for ob_detected in ob_detected:
        distance = np.linalg.norm(particles[:, 0:2] - ob_detected, axis=1) # 取絕對值,按行向量處理
        weights *= scipy.stats.norm.pdf(distance,R,dt_total[i]) # 取密度??
        i += 1

    weights += 1.e-300
    weights /= sum(weights) # normalize   

def neff(weights):
    # 1 / sum of weight^2
	return 1. / np.sum(np.square(weights))

def simple_resample(particles, weights):
	N = len(particles)
	cumulative_sum = np.cumsum(weights) # ex:a=[1,2,3], np.cumsum(a)=[1,3,6]
	cumulative_sum[-1] = 1.	# avoid round-off error
	indexes = np.searchsorted(cumulative_sum, np.random.random(N)) # ex:a=[1,3,5], np.searchsorted(a,2)=1 default'left', indexes is an array

	# resample according to indexes
	particles[:] = particles[indexes]
	weights[:] = weights[indexes]
	weights /= np.sum(weights) # normalize  

def estimate(particles, weights):
	# returns mean and variance of the weighted particles

	pos = particles[:, 0:2]
	mean = np.average(pos, weights=weights, axis=0)
	var  = np.average((pos - mean)**2, weights=weights, axis=0)
	return mean, var

def main(N,scope):
    # 起點終點移動路徑
    sx = 10
    sy = 10
    gx = 27
    gy = 33
    step = np.array([[11,12],
                     [11,14],
                     [12,14],
                     [11,16],
                     [10,18],
                     [12,19],
                     [12,21],
                     [13,23],
                     [14,25],
                     [12,26],
                     [13,27],
                     [13,29],
                     [12,31],
                     [12,33],
                     [11,34],
                     [12,36],
                     [13,38],
                     [13,40],
                     [13,42],
                     [15,43],
                     [17,44],
                     [19,43],
                     [21,42]])

    # 畫地圖
    ob = []
    ob_x, ob_y = [], []
    for i in range(0,60):
        ob_x.append(i)
        ob_y.append(0)
        ob.append([i,0])
    for i in range(0,60):
        ob_x.append(60)
        ob_y.append(i)
        ob.append([60,i])
    for i in range(0,61):
        ob_x.append(i)
        ob_y.append(60)
        ob.append([i,60])
    for i in range(0,61):
        ob_x.append(0)
        ob_y.append(i)
        ob.append([0,i])
    for i in range(0,43):
        ob_x.append(22)
        ob_y.append(i)
        ob.append([20,i])
    for i in range(0,36):
        ob_x.append(43)
        ob_y.append(60-i)
        ob.append([40,60-i])
    for i in range(0,9):
        ob_x.append(i)
        ob_y.append(40)
        ob.append([i,40])
    for i in range(0,7):
        ob_x.append(i)
        ob_y.append(20)
        ob.append([i,20])

    # particles包括0:x 1:y 2:dir, 權重另外放weight
    weights = np.zeros(N)
    particles = init((0, scope), (0, scope), (0, 2*np.pi), N)
    # plt.scatter(particles[:, 0], particles[:, 1],s=0.5)

    plt.plot(ob_x, ob_y, ".k")
    st = plt.plot(sx, sy, "^", label="start", markersize=8)
    gl = plt.plot(gx, gy, "v", label="goal", markersize=8)
    plt.plot(particles[:, 0], particles[:, 1], ".b", markersize=4)
    plt.grid(True)
    plt.axis("equal")
    plt.legend(loc='upper right')
    plt.waitforbuttonpress()

    for i in range(len(step)):
        robot_pos = step[i] # 當前位置
        sensor_std_err = 0.2
        # if(i >= 2):
        #     der0 = -math.atan2((step[i, 1] - step[i-2, 1]), (step[i, 0] - step[i-2, 0])) - math.atan2((step[i, 1] - step[i-1, 1]), (step[i, 0] - step[i-1, 0]))
        # else:
        #     der0 = 0
        if(i>0):
            der1 = ((step[i, 0] - step[i-1, 0])**2 + (step[i, 1] - step[i-1, 1])**2)**0.5
        else:
            der1 = 0
        der = np.array([0, 1])

        # 使用sensor找出範圍內障礙物
        ob_detected =[]
        for x in range(robot_pos[0] - 15, robot_pos[0] + 21):
            for y in range(robot_pos[1], robot_pos[1] + 21):
                for j in range(len(ob)):
                    if([x,y] == ob[j]):
                        ob_detected.append([x,y])
        ob_detected = np.array(ob_detected)
        if(len(ob_detected) == 0):
            print("no obstacle detected!")
            continue
        # 取機器人和障礙物距離
        dt_total = np.linalg.norm(ob_detected - robot_pos, axis=1) + (np.random.randn(len(ob_detected)) * sensor_std_err)
        # 移動點 input:輸入, noise:噪聲
        predict(particles, input=der, noise=(.2, .05))
        # 更新權重
        update(particles, weights, dt_total=dt_total, R=sensor_std_err, ob_detected=ob_detected)
        # 多項式重採樣(multinomial resampling)
        if (neff(weights) < N/2):
            simple_resample(particles, weights)

        mean, var = estimate(particles, weights)
        xs = []
        xs.append(np.rint(mean))

        plt.cla()
        xs = np.array(xs)
        fp = plt.plot(xs[:, 0], xs[:, 1], "or", label="predict pos")
        print('estimated position and variance:\n\t', mean, var)
        ap = plt.plot(step[i, 0], step[i, 1], "ob",label="actual pos")
        plt.plot(ob_x, ob_y, ".k")
        plt.plot(sx, sy, "^")
        plt.plot(gx, gy, "v")
        plt.plot(particles[:, 0], particles[:, 1], ".b", markersize=2)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.axis("equal")
        plt.waitforbuttonpress()

    plt.show()

if __name__ == '__main__':
    main(N=600, scope=60)