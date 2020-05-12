#!/usr/bin/env python3
import numpy as np
import scipy.stats
from numpy.random import uniform,randn,random
import matplotlib.pyplot as plt

import time



def create_uniform_particles(x_range, y_range, hdg_range, N):
	# 生出初始5000個點
	particles = np.empty([N, 3]) # N個模擬點,每個點存x,y,方向
	particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N) # 從頭到尾隨機產生範圍內x值
	particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
	particles[:, 2] = np.random.uniform(hdg_range[0], hdg_range[1], size=N)
	particles[:, 2] %= 2 * np.pi # 轉成角度
	return particles

    
def predict(particles, u, std, dt=1.):
	# 改點的位置
	# 輸入u(轉向, 速度)
	# 干擾資料Q (std 轉向, std 速度)

	N = len(particles)
	# update heading
	particles[:, 2] += u[0] + (np.random.randn(N) * std[0]) #***
	particles[:, 2] %= 2 * np.pi

	# move in the (noisy) commanded direction
	dist = (u[1] * dt) + (np.random.randn(N) * std[1])
	particles[:, 0] += np.cos(particles[:, 2]) * dist	# x項
	particles[:, 1] += np.sin(particles[:, 2]) * dist	# y項
    
    
def update(particles, weights, z, R, landmarks):
	#更新粒子權重
	weights.fill(1.)	  # weights都填上1
	i = 0
	for landmark in landmarks:
		distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)	# 求範數
		weights *= scipy.stats.norm(distance, R).pdf(z[i])
		i = i + 1

	weights += 1.e-300      # avoid round-off to zero
	weights /= sum(weights) # normalize    
    
    
def estimate(particles, weights):
	# returns mean and variance of the weighted particles

	pos = particles[:, 0:2]
	mean = np.average(pos, weights=weights, axis=0)	# mean = mu
	var  = np.average((pos - mean)**2, weights=weights, axis=0)
	return mean, var    
    
    
def neff(weights):
	return 1. / np.sum(np.square(weights))    
    
    
def simple_resample(particles, weights):
	#對樣本做多項式重採樣(multinomial resampling),
	# k = 1 :
	# for i = 1 : N
	# 	while(wc(k) < x(i))	(粒子權重累積函數wc, [0,1]上均勻分佈隨機數x)
	# 	k = k + 1;
	# 	end
	# 	ind(i) = k;	(第k粒子重採樣後複製在i位置)
	# end
	N = len(particles)
	cumulative_sum = np.cumsum(weights) 	# ex:a=[1,2,3], np.cumsum(a)=[1,3,6]
	cumulative_sum[-1] = 1.	# avoid round-off error
	indexes = np.searchsorted(cumulative_sum, random(N))	# ex:a=[1,3,5], np.searchsorted(a,2)=1 default'left', indexes is an array

	# resample according to indexes
	particles[:] = particles[indexes]
	weights[:] = weights[indexes]
	weights /= np.sum(weights) # normalize  
    
def main(N, iters=18, sensor_std_err=0.1, xlim=(0, 20), ylim=(0, 20)):    
	landmarks = np.array([[-3, 2], [3, 12], [10,15], [19,17]]) 
	NL = len(landmarks)
	
	# create particles and weights
	particles = create_uniform_particles((0,20), (0,20), (0, 2*np.pi), N)
	weights = np.zeros(N)
	# plt.scatter(particles[:, 0], particles[:, 1],s=0.5)
	# time.sleep(5)
	
	xs = []   # estimated values
	robot_pos = np.array([0., 0.])
	
	for x in range(iters):
		robot_pos += (1, 1) 
		
		# distance from robot to each landmark
		zs = np.linalg.norm(landmarks - robot_pos, axis=1) + (randn(NL) * sensor_std_err)
		
		# move particles forward to (x+1, x+1)
		predict(particles, u=(0.00, 1.414), std=(.25, .055))
		
		# incorporate measurements
		update(particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks)
		
		# resample if too few effective particles
		if neff(weights) < N/2:
			simple_resample(particles, weights)
		
		# Computing the State Estimate
		mu, var = estimate(particles, weights)
		xs.append(mu)
	
	xs = np.array(xs)
	plt.plot(np.arange(iters+1)) # ,'k+'
	plt.plot(xs[:, 0], xs[:, 1]) # 'r.'
	plt.scatter(landmarks[:,0],landmarks[:,1],alpha=4,marker='o',c=randn(4),s=100) # plot landmarks
	plt.legend( ['real path','sim path'], loc=4, numpoints=1)
	plt.xlim([-10,20])
	plt.ylim([-10,22])
	print('estimated position and variance:\n\t', mu, var)
	plt.show()
    

if __name__ == '__main__':    
	main(N=10000)