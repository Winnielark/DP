# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random
 
if __name__=='__main__':
    plt.rcParams["figure.figsize"]=(14,8)
    model_num=5#生成的模型个数

    n_observations=50#样本点的个数
    xs=np.linspace(-3,3, n_observations)
    ys=np.sin(xs)

    n_samples=40#训练集的点个数
    fed=[]#记录预测结果，行是每一次的预测结果，列是不同训练集
    for j in range(model_num):
		#随机选取训练集
        xs_index=list(range(n_observations))
        xs_index = random.sample(xs_index, n_samples)
        xs_index=np.array(xs_index)
        xd=xs[xs_index]
        yd=ys[xs_index]

        X=tf.placeholder(tf.float32,name="X")
        Y=tf.placeholder(tf.float32,name="Y")
        #初始化参数和权重
        W=tf.Variable(tf.random_normal([1]),name="weight")
        b=tf.Variable(tf.random_normal([1]),name="bias")
        #计算预测结果
        Y_pred=tf.add(tf.multiply(X,W),b)
        W_2=tf.Variable(tf.random_normal([1]),name="weight_2")
        Y_pred=tf.add(tf.multiply(tf.pow(X,2),W_2),Y_pred)
        W_3=tf.Variable(tf.random_normal([1]),name='weight_3')
        Y_pred=tf.add(tf.multiply(tf.pow(X,3),W_3),Y_pred)
        W_4=tf.Variable(tf.random_normal([1]),name='weight_4')
        Y_pred=tf.add(tf.multiply(tf.pow(X,4),W_4),Y_pred)
        
		#计算损失函数值
        sample_num=xs.shape[0]
        loss=tf.reduce_sum(tf.pow(Y_pred-Y,2))/sample_num

        #初始化optimizer
        learning_rate=0.01
        optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        #指定迭代次数，在session里执行graph

        with  tf.Session() as sess:
            #初始化所有变量
            sess.run(tf.global_variables_initializer()) 
            #训练模型
            for  i  in range(400):
                for x,y in zip(xd,yd):
                #通过feed_dict把数据装进去, optimizer和loss为两个节点，但是我只要loss的输出结果.
                    #o,l=sess.run([optimizer,loss],feed_dict={X:x,Y:y})  
                    sess.run([optimizer],feed_dict={X:x,Y:y})

            W,W_2,W_3,W_4,b=sess.run([W,W_2,W_3,W_4,b])
            # W,W_2,W_3,b=sess.run([W,W_2,W_3,b])
            # W,W_2,b=sess.run([W,W_2,b])
            # W,b=sess.run([W,b])

        print('W:'+str(W[0]))
        print('W_2:'+str(W_2[0]))
        print('W_3:'+str(W_3[0]))
        print('W_4:'+str(W_4[0]))
        print('b:'+str(b[0]))
        
        y=xs*W[0]+np.power(xs,2)*W_2[0]+np.power(xs,3)*W_3[0]+np.power(xs,4)*W_4[0]+b[0]
        # y=xs*W[0]+np.power(xs,2)*W_2[0]+np.power(xs,3)*W_3[0]+b[0]
        # y=xs*W[0]+np.power(xs,2)*W_2[0]+b[0]
        # y=xs*W[0]+b[0]
        plt.plot(xs,y,'r',label='Predicted data')
        if j==0:
            fed=y
        else:
            fed=np.row_stack((fed,y))


    efd=np.mean(fed, axis=0)
    bias=(efd-ys).sum()

    variance=np.var(fed,axis=0)
    variance=variance.sum()

    print('bias:'+str(bias))
    print('variance:'+str(variance))
    plt.scatter(xs,ys)#绘图
    plt.plot(xs,ys,'bo',label='Real data')

    plt.legend()
    plt.show()




   
    


