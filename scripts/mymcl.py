'''

made by yang xing yeeeeee
1.初始化,在地圖上隨機灑點(最小單位1 int)
2.取sensor model觀測到的數據(19條laser, 長30int)
3.移動預測點(掉到地圖外的還沒處理)
4.權重更新(貝氏濾波)(取得被laser偵測倒的障礙物位置)
5.重新取樣
6.重複2~5直到結束

'''
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math

def map_plot( ob_x, ob_y, sx, sy, gx, gy, particles ) :
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

def init( x_range, y_range, hdg_range, N ) :
    # 第一列x座標(int), 第二列y座標(int), 第三列弧度(float)
	particles = np.empty([N, 3])
	particles[:, 0] = np.random.randint(x_range[0], x_range[1], size=N)
	particles[:, 1] = np.random.randint(y_range[0], y_range[1], size=N)
	particles[:, 2] = np.random.uniform(hdg_range[0], hdg_range[1], size=N) # 角度
	particles[:, 2] %= 2 * np.pi # 轉成弧度
	return particles

def predict( particles, input, noise, dt=1.0 ) :
    N = len(particles)
    # 更新面向
    particles[:, 2] += input[0] + (np.random.randn(N) * noise[0])
    particles[:, 2] %= 2 * np.pi

    # 更新位置
    dist = (input[1] * dt) + (np.random.randn(N) * noise[1])
    particles[:, 0] += np.rint(np.cos(particles[:, 2]) * dist)
    particles[:, 1] += np.rint(np.sin(particles[:, 2]) * dist)

def find_intersection( p0, p1, p2, p3 ) :
    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]

    denom = s10_x * s32_y - s32_x * s10_y

    if denom == 0 : 
        return None, None
    denom_is_positive = denom > 0

    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]
    s_numer = s10_x * s02_y - s10_y * s02_x

    if (s_numer < 0) == denom_is_positive : 
        return None, None
    t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == denom_is_positive : 
        return None, None
    if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive : 
        return None, None

    t = t_numer / denom
    intersection_point_x = p0[0] + (t * s10_x)
    intersection_point_y = p0[1] + (t * s10_y) 
    return intersection_point_x, intersection_point_y

def update( particles, weights, dist_robot, sensor_std_err, ob_detected ) :
    weights.fill(1.)
    i = 0
    for ob_detected in ob_detected:
        distance = np.linalg.norm(particles[:, 0:2] - ob_detected, axis=1) # 取絕對值,按行向量處理
        weights *= scipy.stats.norm.pdf(distance, sensor_std_err, dist_robot[i]) # 取密度??
        i += 1

    weights += 1.e-300
    weights /= sum(weights) # normalize   

def neff( weights ) :
    # 1 / sum of weight^2
	return 1. / np.sum(np.square(weights)) 

def resample_elite( particles, weights, range ) :
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

def resample_compete( particles, weights, range ) :
    N = len(particles)
    old_particles = particles
    old_weights = weights
    for i in range(N) :
        chosed = np.random.randint(0, N, size=3)
        best = max(weights[chosed[0]], weights[chosed[1]], weights[chosed[2]])
        best_num = 0
        for j in range(3) :
            if best == weights[chosed[j]] :
                best_num = chosed[j]
        particles[i] = old_particles[best_num]
        weights[i] = old_weights[best_num]

def estimate( particles, weights ) :
    max_w = np.argmax(weights)
    mean = particles[max_w]
    # mean = np.average(particles,weights=weights, axis=0)
    var  = np.average((particles - mean)**2, weights=weights, axis=0)
    return mean, var

def main( N,scope ) :
    # 起點終點移動點
    sx = 10
    sy = 10
    gx = 28
    gy = 33
    step = np.array([[11,11],
                     [11,13],
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

    # 建地圖
    ob = []
    ob_x, ob_y = [], []
    for i in range(0, 60) :
        ob_x.append(i)
        ob_y.append(0)
        ob.append([i, 0])
    # ob.append([0, 0])
    # ob.append([60, 0])
    for i in range(0, 60) :
        ob_x.append(60)
        ob_y.append(i)
        ob.append([60, i])
    # ob.append([60, 0])
    # ob.append([60, 60])
    for i in range(0, 61) :
        ob_x.append(i)
        ob_y.append(60)
        ob.append([i, 60])
    # ob.append([0, 60])
    # ob.append([60, 60])
    for i in range(0, 61) :
        ob_x.append(0)
        ob_y.append(i)
        ob.append([0, i])
    # ob.append([0, 0])
    # ob.append([0, 60])
    for i in range(0, 43) :
        ob_x.append(22)
        ob_y.append(i)
        ob.append([22, i])
    # ob.append([22, 0])
    # ob.append([22, 43])
    for i in range(0, 36) :
        ob_x.append(43)
        ob_y.append(60-i)
        ob.append([43, 60-i])
    # ob.append([43, 60])
    # ob.append([43, 24])
    for i in range(1,12):
        ob_x.append(i)
        ob_y.append(40)
        ob.append([i, 40])
    # ob.append([0, 40])
    # ob.append([12, 40])
    for i in range(1,7) :
        ob_x.append(i)
        ob_y.append(20)
        ob.append([i, 20])
    # ob.append([0, 20])
    # ob.append([7, 20])
    for i in range(35,44) :
        ob_x.append(i)
        ob_y.append(24)
        ob.append([i, 24])
    # ob.append([35, 24])
    # ob.append([44, 24])
    for i in range(1,15) :
        ob_x.append(35)
        ob_y.append(i)
        ob.append([35, i])
    # ob.append([35, 0])
    # ob.append([35, 15])

    # 初始化
    # particles 0:x 1:y 2:dir, 權重weight
    weights = np.zeros(N)
    particles = init((1, scope-1), (1, scope-1), (0, 2*np.pi), N)

    map_plot(ob_x, ob_y, sx, sy, gx, gy, particles)
    plt.waitforbuttonpress()

    dir = 0
    for i in range(len(step)) :
        robot_pos = step[i]
        print(robot_pos)
        sensor_std_err = 0.2
        # dir方向, dir_change dt距離
        dir_old = dir
        dt = 0
        if i > 0 :
            dir = math.atan2(step[i, 1] - step[i-1, 1], step[i, 0] - step[i-1, 0])
        elif i == 0 :
            dir = np.pi/4
        dir_change = dir - dir_old
        if i > 0 :
            dt = np.sqrt((step[i, 0] - step[i-1, 0])**2 + (step[i, 1] - step[i-1, 1])**2)
        else :
            dt = 1.414
        print(dir)
        print(dt)
        u = np.array([dir_change, dt])

        ob_detected = []
        for x in range(robot_pos[0] - 15, robot_pos[0] + 16) :
            for y in range(robot_pos[1] - 15, robot_pos[1] + 16) :
                for j in range(len(ob)) :
                    if([x,y] == ob[j]) :
                        ob_detected.append([x,y])
        if len(ob_detected) == 0 :
            print("no obstacle detected around robot!")
            continue

        # 模擬每10度角放置一雷射模型(0~180),雷射長度30int
        # ob_detected = []
        # for j in range(19) :
        #     isdetect = 0
        #     min_x = 0
        #     min_y = 0
        #     intersection_list = []
        #     laser_scan = [ 30 * math.cos(j * 10 * np.pi / 180), 30 * math.sin(j * 10 * np.pi / 180) ]
        #     for k in range(int(len(ob)/2)) :
        #         intersection_x, intersection_y = find_intersection( ob[2 * k], ob[2 * k + 1], robot_pos, laser_scan )
        #         if intersection_x != None :
        #             intersection_list.append([intersection_x, intersection_y])
        #             isdetect = 1
        #     intersection_list = np.array(intersection_list)
        #     if len(intersection_list) > 1 :
        #         min = scope * 10
        #         for x in range(len( intersection_list)) :
        #             length_numx = np.sqrt((intersection_list[x, 0] - robot_pos[0]) ** 2 + (intersection_list[x, 1] - robot_pos[1]) ** 2)
        #             if length_numx < min :
        #                 min = length_numx
        #                 min_x = intersection_list[x, 0]
        #                 min_y = intersection_list[x, 1]
        #     if isdetect == 0 :
        #         ob_detected.append(laser_scan)
        #     elif len(intersection_list) > 1 :
        #         ob_detected.append([min_x, min_y])
        #     else :
        #         ob_detected.append([intersection_list[0, 0], intersection_list[0, 1]])

        dist_robot = np.linalg.norm(ob_detected - robot_pos, axis=1) + (np.random.randn(len(ob_detected)) * sensor_std_err)
        # 移動點 input:輸入, noise:噪聲
        predict(particles, input=u, noise=(.2, .02))
        # 更新權重
        update(particles, weights, dist_robot, sensor_std_err, ob_detected)
        # 重取樣
        print("neff(weights) is %2f" %(neff(weights)))
        if neff(weights) < N/2 :
            resample_compete(particles, weights, range)
            # resample_elite(particles, weights, range)

        mean, var = estimate(particles, weights)
        xs = []
        xs.append(np.rint(mean))

        plt.clf()
        xs = np.array(xs)
        print('estimated position and variance:\n\t', mean, var)
        map_plot(ob_x, ob_y, sx, sy, gx, gy, particles)
        for j in range(N) :
            sim_x, sim_y = (np.array([particles[j, 0], particles[j, 1]]) + np.array([np.cos(particles[j, 2]), np.sin(particles[j, 2])]) * 0.5)
            plt.plot([particles[j, 0], sim_x], [particles[j, 1], sim_y], "-k")
        fp = plt.plot(xs[:, 0], xs[:, 1], "or", label="predict pos", markersize=12)
        ap = plt.plot(step[i, 0], step[i, 1], "ob",label="actual pos", markersize=12)
        robot_x, robot_y = (np.array([robot_pos[0], robot_pos[1]]) + np.array([np.cos(dir), np.sin(dir)]) * 1.7)
        plt.plot([robot_pos[0], robot_x], [robot_pos[1], robot_y], "-k", linewidth = 2.5)
        plt.waitforbuttonpress()

    plt.show()

if __name__ == '__main__' :
    main( N=600, scope=60 )