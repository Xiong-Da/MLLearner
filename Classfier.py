import random
import numpy as np
import tensorflow as tf


class Trainer:
    def __init__(self,image,param):
        self.image=image
        self.param=param
        self.shape=image.shape
        self.totalCord=None
        self.totalLabel=None

        self.sess=None
        self.inputCord=None
        self.inputLabel=None
        self.output=None
        self.train_step=None
        self.accuracy=None

        self.layers=[]

        self.initData()
        self.initNN()

    def initData(self):
        cord=[]
        label=[]
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                cord.append([float(x),float(y)])
                if self.image[x][y]>0:
                    label.append([1.0,0.0])
                else:
                    label.append([0.0,1.0])
        self.totalCord=np.array(cord)
        self.totalLabel=np.array(label)

    def getBatch(self,num):
        cord=[]
        label=[]
        for i in range(num):
            x=random.randint(0,self.shape[0]-1)
            y=random.randint(0,self.shape[1]-1)
            cord.append([float(x),float(y)])
            label.append(self.totalLabel[x*self.shape[1]+y])
        return (np.array(cord),np.array(label))

    def getAll(self):
        return (self.totalCord,self.totalLabel)

    def initNN(self):
        self.sess=tf.InteractiveSession()
        self.inputCord=tf.placeholder(tf.float32,[None,2])
        self.inputLabel=tf.placeholder(tf.float32,[None,2])

        self.layers.clear()

        lastInput=self.inputCord
        lastNum=2
        for num in self.param:
            w=tf.Variable(tf.truncated_normal([lastNum,num],stddev=0.1))
            b=tf.Variable(tf.zeros([num]))
            lastInput=tf.nn.relu(tf.matmul(lastInput,w)+b)
            lastNum=num

            self.layers.append(w)

        w = tf.Variable(tf.truncated_normal([lastNum, 2], stddev=0.1))
        b = tf.Variable(tf.zeros([2]))
        y=tf.matmul(lastInput,w)+b
        self.output=tf.nn.softmax(y)

#don't use this method,it may couse NAN problem
#        cross_entropy=tf.reduce_mean(-tf.reduce_sum(self.inputLabel*tf.log(self.output),
#                                                    reduction_indices=[1]))

        cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.inputLabel,logits=y
        ))

        self.train_step=tf.train.AdamOptimizer().minimize(cross_entropy)

        correct=tf.equal(tf.argmax(self.output,1),tf.argmax(self.inputLabel,1))
        self.accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

        tf.global_variables_initializer().run()

    def train(self,num):
        cord,label=self.getBatch(num)
        self.train_step.run({self.inputCord:cord,self.inputLabel:label})

    def getAccuracy(self):
        cord,label=self.getAll()
        return self.sess.run([self.accuracy],{self.inputCord:cord,self.inputLabel:label})[0]

    def getNNState(self):
        cord, label = self.getAll()
        result=self.sess.run([self.output,self.accuracy]+self.layers
                             ,{self.inputCord:cord,self.inputLabel:label})

        outputPatten=result[0]
        accuracy=result[1]

    #    image=np.zeros(self.shape,np.uint8)

    #    for i in range(len(cord)):
    #        if outputPatten[i][0]>outputPatten[i][1]:
    #            image[int(cord[i][0])][int(cord[i][1])]=255

        image=tf.reshape(tf.cast(tf.argmin(outputPatten,1),tf.uint8)*255,self.shape)

    #    for i in range(2,len(result)):
    #        print("\n\nw"+str(i-1)+":")
    #        print(result[i])

        return image.eval(),accuracy


import threading
curThread=None
needStop=False

def threadFun(image,param,window):
    global needStop

    nnHandler = Trainer(image, param)

    for i in range(10000000):
        if needStop:return

        nnHandler.train(50)

        if i%100==0:
            outputPatten,accuracy=nnHandler.getNNState()
            message="iter:"+str(i)+"  accuracy:"+str(accuracy)

            window.setOutputPatten(outputPatten,message)

def startNN(image,param,window):
    global curThread
    global needStop
    needStop = True

    thread=threading.Thread(target=threadFun,args=(image,param,window))
    thread.setDaemon(True)

    if curThread != None:
        curThread.join()
    curThread = thread
    needStop = False

    thread.start()