#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:14:26 2023

@author: ghosh.244a
"""
###Experiment for A paradoxical CMDP
#import gym
import math
import numpy as np
from scipy.stats import bernoulli

K=10000;
state_sp_x=2;
state_sp_th=1;
state_sp_x_dot=1;
state_sp_theta_dot=1;
success=0;

A=2;
H=10;
beta=5*math.sqrt(math.log((K+1)))
eta=3/(math.sqrt(K*H))
alpha=math.sqrt(K)/(H);
#reach_goal=0;
Y=0
b=5
S=(state_sp_x)*(state_sp_th)*(state_sp_x_dot)*(state_sp_theta_dot);

#print(f'location {location}, speed {speed}')

Q_val_r=np.zeros([S,H+1,A])
Q_val_g=np.zeros([S,H+1,A])
V_r= np.zeros([S,H+1])
V_g= np.zeros([S,H+1])
pol=(1/A)*np.ones([S,H,A])
n_h=np.ones([S,H,A],dtype=int)
next_st={}
r_h={}
g_h={}
tot_rew=np.zeros([K,1]);
no_re_loop=np.zeros([K,1]);
tot_uti=np.zeros([K,1])
#frozen_lake=gym.make('FrozenLake-v1', desc=None, map_name="4x4",is_slippery=False)

for k in range(K):
    Y=0
    #observation = frozen_lake.reset()
    #print('state',observation)
    #x=observation[0];
    reach_goal=0
    x1=0;
    for step in range(H):
        #x1=state_cod(location,speed,theta,omega)
        expe=np.random.multinomial(1,pol[x1,step,:])
        action_ar=np.nonzero(expe)
        action=action_ar[0][0]
        if action==0 and x1==0:
            x_11=0
            reward=1
            utility=0
        elif action==1 and x1==0:
            x_11=1
            reward=0
            utility=1
        elif action==0 and x1==1:
            x_11=1
            reward=0
            utility=1
        elif action==1 and x1==1:
            x_11=0
            reward=0
            utility=0
        #print('state and action',x1,action)
        #prb=pol[x1,step,1];
        #action=bernoulli.rvs(prb)
        #observation, reward, hj, done, _ = frozen_lake.step(action)
        #if hj==True:
            
            #break
        #location=observation[0];
        #speed=observation[1];
        #theta=observation[2];
        #omega=observation[3]
        #if reach_goal ==1:
           # x_11=x1
        #else:
           # x_11=observation
        #print('next state',x_11)    
        #print('reward',reward)
        # if -2.2>=location>=-2.4:
        #     utility=0.0
        # elif 2.2<=location<=2.4:
        #     utility =0.0
        # elif -1.1>=location>-1.3:
        #     utility=0.0;
    
        # elif 1.1<=location<=1.3:
        #     utility=0.0
        # else:
        #     utility=1.0
        # if x_11==19 or x_11==29 or x_11==35 or x_11==41 or x_11==42 or x_11==46 or x_11==49 or x_11==52 or x_11==54 or x_11==59:  
        #     utility=0
        # elif x_11==8 or x_11==16 or x_11==24 or x_11==7 or x_11==15 or x_11==23 or x_11==31 or x_11==39:
        #     utility=0
        # else:
        #     utility=1
        #print('termination',hj)
        #if reward==1:
           # reach_goal=1;
        if reach_goal==1:
            tot_rew[k]=tot_rew[k]+1
            tot_uti[k]=tot_uti[k]+1
            r_h[x1,step,n_h[x1,step,action]-1,action]=1
            g_h[x1,step,n_h[x1,step,action]-1,action]=1
            next_st[x1,step,n_h[x1,step,action]-1,action]=x1
        else:
                tot_rew[k]=tot_rew[k]+reward
                tot_uti[k]=tot_uti[k]+utility
                r_h[x1,step,n_h[x1,step,action]-1,action]=reward
                g_h[x1,step,n_h[x1,step,action]-1,action]=utility
                next_st[x1,step,n_h[x1,step,action]-1,action]=x_11
        n_h[x1,step,action]+=1;
        #tot_rew[k]=tot_rew[k]+reward
        #tot_uti[k]=tot_uti[k]+utility
        #print(f'location {location}, speed {speed},theta {theta}, omega {omega}')
        #print('reward, and utility are', reward,utility,theta)
        x1=x_11
    while  Y<=5:    
     for h in reversed(range(H)):    
        for states in range(S):
           qu_s=0
           qu_exp=0
           for action in range (A):
              phi=beta*(1/n_h[states,h,action])
              #phi=0
              qu_r=0
              qu_g=0
              
              for j in range(n_h[states,h,action]-1):
                  qu_r+=(1/(n_h[states,h,action]))*(r_h[states,h,j,action]+V_r[next_st[states,h,j,action],h+1]);
                  qu_g+=(1/(n_h[states,h,action]))*(g_h[states,h,j,action]+V_g[next_st[states,h,j,action],h+1]);
              Q_val_r[states,h,action]=min(qu_r+phi,H-h)
              Q_val_g[states,h,action]=min(qu_g+phi,(H-h))
              qu_exp+=math.exp(alpha*(Q_val_r[states,h,action]+Y*Q_val_g[states,h,action]))
           # if alpha*(Q_val_r[states,h,0]+Y*Q_val_g[states,h,0]-Q_val_r[states,h,1]-Y*Q_val_g[states,h,1])>709:
           #     pol[states,h,0]=1;
           #     pol[states,h,1]=0;
           # elif alpha*(Q_val_r[states,h,1]+Y*Q_val_g[states,h,1]-Q_val_r[states,h,0]-Y*Q_val_g[states,h,0])>709:
           #     pol[states,h,1]=1;
           #     pol[states,h,0]=0;
           # else:
           #   norm0=math.exp(alpha*(Q_val_r[states,h,0]+Y*Q_val_g[states,h,0]-Q_val_r[states,h,1]-Y*Q_val_g[states,h,1]))
           #   norm1=math.exp(alpha*(Q_val_r[states,h,1]+Y*Q_val_g[states,h,1]-Q_val_r[states,h,0]-Y*Q_val_g[states,h,0]))
           #   pol[states,h,0]=1/(1+norm1)
           #   pol[states,h,1]=1/(1+norm0)
           for action in range(A):
               if alpha*(Q_val_r[states,h,action]+Y*Q_val_g[states,h,action])>709:
                   pol[states,h,action]=1;
               else:
                pol[states,h,action]=math.exp(alpha*(Q_val_r[states,h,action]+Y*Q_val_g[states,h,action]))/qu_exp;
           v_t=0
           v_g=0
           for action in range (A):
               v_t+=pol[states,h,action]*Q_val_r[states,h,action]
               v_g+=pol[states,h,action]*Q_val_g[states,h,action]
           V_r[states,h]=v_t
           V_g[states,h]=v_g
     if V_g[0,0]>=5:    
        break
     Y=Y+eta
     no_re_loop[k]=no_re_loop[k]+1;
    if tot_rew[k]>0:
       success+=1;
    
    #Y=max(min(Y+eta*(b-(V_g[0,0])),5),0)  
    print('tot_rew, tot_uti, iteration no.',tot_rew[k],tot_uti[k],k)  



