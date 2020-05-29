'''

made by yang xing yeeeeee
1.初始化,在地圖上隨機灑點(最小單位1 int)
2.移動預測點(掉到地圖外的還沒處理)
3.取每個點和附近障礙勿內距離(x:+-15,+15~0)
4.權重更新(貝氏濾波)(取得被laser偵測倒的障礙物位置)
5.重新取樣
6.重複2~5直到結束

'''

# useful plt function:
#     plt.waitforbuttonpress()

# trouble 1:點不能灑在障礙上

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math

def map_plot(ob_x, ob_y, sx, sy, gx, gy, particles):
    plt.plot([0, 0], [0, 60], color = 'k', linewidth = 5)
    plt.plot([0, 60], [0, 0], color = 'k', linewidth = 5)
    plt.plot([0, 60], [60, 60], color = 'k', linewidth = 5)
    plt.plot([60, 60], [0, 60], color = 'k', linewidth = 5)
    plt.plot([0, 7], [20, 20], color = 'k', linewidth = 5)
    plt.plot([0, 12], [40, 40], color = 'k', linewidth = 5)
    plt.plot([22, 22], [0, 43], color = 'k', linewidth = 5)
    plt.plot([35, 35], [0, 15], color = 'k', linewidth = 5)
    plt.plot([35, 43], [24, 24], color = 'k', linewidth = 5)
    plt.plot([43, 43], [24, 60], color = 'k', linewidth = 5)
    plt.plot(ob_x, ob_y, ".k")
    plt.plot(particles[:, 0], particles[:, 1], ".b", markersize=4)
    st = plt.plot(sx, sy, "^", label="start", markersize=8)
    gl = plt.plot(gx, gy, "v", label="goal", markersize=8)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis("equal")

def init(x_range, y_range, hdg_range, N):
    # 第一列x座標(float), 第二列y座標(float), 第三列弧度(float)
	particles = np.empty([N, 3])
	particles[:, 0] = np.random.randint(x_range[0], x_range[1], size=N)
	particles[:, 1] = np.random.randint(y_range[0], y_range[1], size=N)
	particles[:, 2] = np.random.uniform(hdg_range[0], hdg_range[1], size=N) # 角度
	particles[:, 2] %= 2 * np.pi # 轉成弧度
	return particles

def predict(particles, input, noise, dt=1.0):
    N = len(particles)
    # 更新面向
    particles[:, 2] += input[0] + (np.random.randn(N) * noise[0])
    particles[:, 2] %= 2 * np.pi

    # 更新位置
    dist = (input[1] * dt) + (np.random.randn(N) * noise[1])
    particles[:, 0] += np.rint(np.cos(particles[:, 2]) * dist)
    particles[:, 1] += np.rint(np.sin(particles[:, 2]) * dist)

def update(particles, weights, dist_robot, sensor_std_err, ob_detected):
    weights.fill(1.)
    i = 0
    for ob_detected in ob_detected:
        distance = np.linalg.norm(particles[:, 0:2] - ob_detected, axis=1) # 取絕對值,按行向量處理
        weights *= scipy.stats.norm.pdf(distance, sensor_std_err, dist_robot[i]) # 取密度??
        i += 1

    weights += 1.e-300
    weights /= sum(weights) # normalize   

def neff(weights):
    # 1 / sum of weight^2
	return 1. / np.sum(np.square(weights))

# def simple_resample(particles, weights):
# 	N = len(particles)
# 	cumulative_sum = np.cumsum(weights) # ex:a=[1,2,3], np.cumsum(a)=[1,3,6]
# 	cumulative_sum[-1] = 1.	# avoid round-off error
# 	indexes = np.searchsorted(cumulative_sum, np.random.random(N)) 
    
# 	# resample according to indexes
# 	particles[:] = particles[indexes]
# 	weights[:] = weights[indexes]
# 	weights /= np.sum(weights) # normalize 

def resample(particles, weights, range):
    N = len(particles)
    valid_particle = np.argpartition(weights, -int(N/15))[-int(N/15):]
    for i in range(N) :
        valid_index = np.random.choice(valid_particle)
        for j in range(len(valid_particle)) :
            if i != valid_particle[j] :
                particles[i, 0] = np.random.randint(low = int(particles[valid_index, 0]) - 1, high = int(particles[valid_index, 0]) + 2, size=1)
                particles[i, 1] = np.random.randint(low = int(particles[valid_index, 1]) - 1, high = int(particles[valid_index, 1]) + 2, size=1)
                particles[i, 2] = particles[valid_index, 2]
                weights[i] = weights[valid_index]

def estimate(particles, weights):
    # N = len(particles)
    # index = np.argpartition(weights, -int(N/20))[-int(N/20):]
    # valid_list = np.empty([(N//20), 2])
    # for i in range(len(index)) :
    #     valid_list[i] = [particles[index[i], 0], particles[index[i], 1]]
    # valid_weight = np.empty([(N//20), 1])
    # for i in range(len(index)) :
    #     valid_weight[i] = [weights[index[i]]]
    mean = np.average(particles,weights=weights, axis=0)
    var  = np.average((particles - mean)**2, weights=weights, axis=0)
    return mean, var

def main(N,scope):
    # 起點終點移動路徑
    sx = 10
    sy = 10
    gx = 28
    gy = 33
    step = np.array([[11,11],
                     [12,12],
                     [12,14],
                     [12,16],
                     [12,18],
                     [13,19],
                     [13,21],
                     [13,23],
                     [14,24],
                     [14,26],
                     [14,28],
                     [14,30],
                     [14,32],
                     [14,34],
                     [14,36],
                     [14,38],
                     [15,39],
                     [15,41],
                     [16,42],
                     [17,44],
                     [19,44],
                     [20,45],
                     [21,46],
                     [23,46],
                     [24,45],
                     [25,44],
                     [26,43],
                     [27,42],
                     [29,42],
                     [29,40],
                     [29,38],
                     [29,36],
                     [29,34],
                     [28,33],])

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
        ob.append([22,i])
    for i in range(0,36):
        ob_x.append(43)
        ob_y.append(60-i)
        ob.append([43,60-i])
    for i in range(1,12):
        ob_x.append(i)
        ob_y.append(40)
        ob.append([i,40])
    for i in range(1,7):
        ob_x.append(i)
        ob_y.append(20)
        ob.append([i,20])
    for i in range(35,44):
        ob_x.append(i)
        ob_y.append(24)
        ob.append([i,24])
    for i in range(1,15):
        ob_x.append(35)
        ob_y.append(i)
        ob.append([35,i])

    # particles包括0:x 1:y 2:dir, 權重放weight
    weights = np.zeros(N)
    particles = init((1, scope-1), (1, scope-1), (0, 2*np.pi), N)

    map_plot(ob_x, ob_y, sx, sy, gx, gy, particles)
    plt.waitforbuttonpress()

    for i in range(len(step)):
        robot_pos = step[i] # 當前位置
        sensor_std_err = 0.2
        # if i >= 2 :
        #     der0 = -math.atan2((step[i, 1] - step[i-2, 1]), (step[i, 0] - step[i-2, 0])) - math.atan2((step[i, 1] - step[i-1, 1]), (step[i, 0] - step[i-1, 0]))
        # else :
        #     der0 = 0
        if i > 0 :
            der1 = ((step[i, 0] - step[i-1, 0])**2 + (step[i, 1] - step[i-1, 1])**2)**0.5
        else:
            der1 = 0
        u = np.array([0, 1])

        # 模擬sensor找出個點附近障礙物放入ob_detected
        # 取機器人和每個障礙物距離計算範數dist_total, dist_sim為粒子距離加總, dist_robot為機器人取得的範數
        # dist_sim = []
        # for n in range(N):
        #     ob_detected = []
        #     for x in range(int(particles[n,0]) - 15, int(particles[n,0]) + 16):
        #         for y in range(int(particles[n,1]) - 15, int(particles[n,1]) + 16):
        #             for j in range(len(ob)):
        #                 if([x,y] == ob[j]):
        #                     ob_detected.append([x,y])
        #     if(len(ob_detected) == 0):
        #         print("no obstacle detected in particle%d!" %(n))
        #         continue
        #     ob_detected = np.array(ob_detected)
        #     ob_detected[:,0] -= int(particles[n,0])
        #     ob_detected[:,1] -= int(particles[n,1])
        #     dist_sim.append([np.linalg.norm(ob_detected, axis=1) + (np.random.randn(len(ob_detected)) * sensor_std_err)])
        ob_detected = []
        for x in range(robot_pos[0] - 15, robot_pos[0] + 16):
            for y in range(robot_pos[1] - 15, robot_pos[1] + 16):
                for j in range(len(ob)):
                    if([x,y] == ob[j]):
                        ob_detected.append([x,y])
        if len(ob_detected) == 0 :
            print("no obstacle detected around robot!")
            continue
        dist_robot = np.linalg.norm(ob_detected - robot_pos, axis=1) + (np.random.randn(len(ob_detected)) * sensor_std_err)
        # 移動點 input:輸入, noise:噪聲
        predict(particles, input=u, noise=(.2, .05))
        # 更新權重
        update(particles, weights, dist_robot, sensor_std_err, ob_detected)
        # 重取樣
        print("neff(weights) is %d" %(neff(weights)))
        if (neff(weights) < N/2):
            resample(particles, weights, range)

        mean, var = estimate(particles, weights)
        xs = []
        xs.append(np.rint(mean))

        plt.clf()
        xs = np.array(xs)
        fp = plt.plot(xs[:, 0], xs[:, 1], "or", label="predict pos", markersize=12)
        print('estimated position and variance:\n\t', mean, var)
        ap = plt.plot(step[i, 0], step[i, 1], "ob",label="actual pos", markersize=12)
        map_plot(ob_x, ob_y, sx, sy, gx, gy, particles)
        plt.waitforbuttonpress()

    plt.show()

if __name__ == '__main__':
    main(N=600, scope=60)