import os
import pickle

from keras.models import load_model, Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import numpy as np

from .env import RLCar

def train():
    env = RLCar()

    try:
        model = load_model('model_porxy.hdf5')
    except:
        model = Sequential()
        model.add(Dense(units=10, input_dim=16))
        model.add(Activation("relu"))
        model.add(Dense(units=12))
        model.add(Activation("relu"))
        model.add(Dense(units=3))
        model.add(Activation("relu"))
        model.compile(optimizer='Adam', loss='categorical_crossentropy')

    # Postavljanje parametara za učenje
    y = .99 #gamma
    e = 0.1 #alpha-learning rate, epsilon greedy constant
    num_episodes = 1500
    num_steps = 300
    #Stvaranje lista koje sadrže ukupnu nagradu i korake po jednoj epizodi
    jList = []
    rList = []
    lList = []
    SPEED = 0.7

    for i in range(num_episodes):
        #Resetiranje okoline i dobijanje prvog zapažanja
        env.reset()
        s, r = env.step([SPEED, SPEED])
        s = s['proxy_sensor'].reshape((1, -1))
        r = r['proxy_sensor']
        Q = model.predict(s)
        a = Q.argmax()
        rAll = 0
        done = False
        loss = 0
        # Q-mreža
        for j in range(num_steps):
            print("Step {} | Action: {} | Reward: {}".format(j, a, r))
            # Odabir akcije po metodi "pohlepnosti" (sa šansom e slučajno odabrane akcije)
            # iz Q-mreže
            Q = model.predict(s)
            a = Q.argmax()
            if np.random.rand(1) < e:
                a = np.random.randint(3)
                print("e = {}. Choosing Random Action: {}".format(e, a))
            #Dobijanje novog stanja i nagrade iz okoliša
            speed = np.zeros(2)
            #Q -> lijevo, desno, naprijed, prekid, natrag
            if a == 0:
                speed[0] = 0
                speed[1] = SPEED
            if a == 1:
                speed[0] = SPEED
                speed[1] = 0
            if a == 2:
                speed[0] = SPEED
                speed[1] = SPEED
            if a == 3:
                speed[0] = 0
                speed[1] = 0

            s_, r_ = env.step(speed)
            s_ = s_['proxy_sensor'].reshape((1, -1))
            r_ = r_['proxy_sensor']
            #Dobijanje Q-vrijednosti hranjenjem novog stanja putem naše mreže
            Q_ = model.predict(s_)
            #Dobijanje maxQ i postavljanje ciljne vrijednosti za odabranu akciju
            maxQ_ = np.max(Q_)
            targetQ = Q
            targetQ[0, a] = r + y * maxQ_
            #Treniranje mreže korištenjem ciljnih i predviđenih Q-vrijednosti
            loss += model.train_on_batch(s, targetQ)
            rAll += r
            s = s_
            r = r_
            if done is True:
                break
        #Smanjenje šanse slučajne akcije dok treniramo model
        e -= 0.001
        jList.append(j)
        rList.append(rAll)
        lList.append(loss)
        print("Episode: " + str(i))
        print("Loss: " + str(loss))
        print("e: " + str(e))
        print("Reward: " + str(rAll))
        pickle.dump({'jList': jList, 'rList': rList, 'lList': lList},
                    open("history_porxy.p", "wb"))
        model.save('model_proxy.hdf5')

    print("Average loss: " + str(sum(lList) / num_episodes))
    print("Average number of steps: " + str(sum(jList) / num_episodes))
    print("Average reward: " + str(sum(rList) / num_episodes))

    plt.plot(rList)
    plt.plot(jList)
    plt.plot(lList)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        train()
    except KeyboardInterrupt:
        print('Exiting.')
