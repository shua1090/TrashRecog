import collections
import multiprocessing as mp
import pickle
import socket
import threading
import time

import cv2
import lz4.block as compressing_engine
import numpy as np
import psutil
import GPUtil
from PIL import Image

from Infer import Model

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def convert(im): return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


def chart_data():
    # function to update the data

    def graph_data(i):
        gpu = GPUtil.getGPUs()[0]

        cpu.popleft()
        ram.popleft()
        gpu_temp.popleft()
        gpu_mem.popleft()

        cpu.append(psutil.cpu_percent(interval=0.01))
        ram.append(psutil.virtual_memory().percent)  # clear axis
        gpu_temp.append(gpu.temperature)
        gpu_mem.append(gpu.memoryUsed)

        ax.cla()
        ax1.cla()
        ax2.cla()
        ax3.cla()

        ax.plot(cpu)
        ax.title.set_text("CPU Load estimate")
        ax.scatter(len(cpu) - 1, cpu[-1])
        ax.text(len(cpu) - 1, cpu[-1] + 2, "{}%".format(cpu[-1]))
        ax.set_ylim(0, 100)

        ax1.plot(ram)
        ax1.title.set_text("RAM Use % estimate")
        ax1.scatter(len(ram) - 1, ram[-1])
        ax1.text(len(ram) - 1, ram[-1] + 2, "{}%".format(ram[-1]))
        ax1.set_ylim(0, 100)  # start collections with zeros

        ax2.set_ylim(20, 60)
        ax2.title.set_text("GPU Temperature (Celsius)")
        ax2.plot(gpu_temp)
        ax2.scatter(len(gpu_temp) - 1, gpu_temp[-1])
        ax2.text(len(gpu_temp) - 1, gpu_temp[-1] + 2, "{}C".format(gpu_temp[-1]))

        ax3.plot(gpu_mem)
        ax3.set_ylim(0, gpu.memoryTotal)  # start collections with zeros
        ax3.title.set_text("VRAM Load estimate")
        ax3.scatter(len(gpu_mem) - 1, gpu_mem[-1])
        ax3.text(len(gpu_mem) - 1, gpu_mem[-1] + 2, "{} MB".format(gpu_mem[-1]))

    cpu = collections.deque(np.zeros(10))
    ram = collections.deque(np.zeros(10))  # define and adjust figure
    gpu_temp = collections.deque(np.zeros(10))
    gpu_mem = collections.deque(np.zeros(10))
    fig = plt.figure(figsize=(16, 6), facecolor='#DEDEDE')

    ax = plt.subplot(141)
    ax1 = plt.subplot(142)
    ax2 = plt.subplot(143)
    ax3 = plt.subplot(144)

    ax.set_facecolor('#DEDEDE')
    ax1.set_facecolor('#DEDEDE')
    ax2.set_facecolor('#DEDEDE')
    ax3.set_facecolor('#DEDEDE')

    plt.gcf().canvas.set_window_title("Benchmark HQ")
    FuncAnimation(fig, graph_data, interval=0)
    plt.show()


def resultSend(q: mp.Queue, sock, lock):
    while True:
        z = q.get(True)
        res = pickle.dumps(z)
        lock.acquire()
        sock.send(res)
        lock.release()
        if z == "END":
            break


def inferAndSend(que, sock):
    m = Model()
    m.load()
    i = 0
    t1 = time.time()

    barray = []

    sendResultLock = mp.Lock()
    sendQueue = mp.Queue(8)

    threading.Thread(target=resultSend, args=(sendQueue, sock, sendResultLock)).start()
    z = threading.Thread(target=resultSend, args=(sendQueue, sock, sendResultLock))
    z.start()

    while True:
        im = que.get(True)
        i += 1

        res = m.predict(im)
        sendQueue.put(res["Prediction"], True)

        cv2ReadyImg = np.array(im)[:, :, ::-1].copy()
        dim = (600, 600)
        cv2ReadyImg = cv2.resize(cv2ReadyImg, dim)

        cv2.putText(cv2ReadyImg,
                    res["Prediction"],
                    (30, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    )

        cv2.putText(cv2ReadyImg,
                    str(round(1 / (time.time() - t1), 2)) + " fps",
                    (450, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    )

        cv2.imshow("Live Video Inference", cv2ReadyImg)

        if cv2.waitKey(1) == 27:  # or len(barray) == 1000:
            print(len(barray) / (barray[-1] - barray[0]), "fps on average over ", len(barray), " images")
            for i in range(10): sendQueue.put("END", True)
            break

        t1 = time.time()
        barray.append(t1)


def clientWorker(sock, lock, que):
    while True:
        lock.acquire()
        recvsize = pickle.loads(sock.recv(8, socket.MSG_WAITALL))
        print(recvsize)
        a = sock.recv(recvsize, socket.MSG_WAITALL)
        lock.release()
        a = pickle.loads(compressing_engine.decompress(a))  # , 2)
        a = convert(a)
        que.put(a, True)


def processWorker(sock, lock, que):
    threading.Thread(target=clientWorker, args=(sock, lock, que)).start()
    t = threading.Thread(target=clientWorker, args=(sock, lock, que))
    t.start()
    t.join()


if __name__ == "__main__":

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create socket
    port = 5000  # Port number, remember to change on both sides if necessary
    s.bind(('', port))
    s.listen(2)
    c, addr = s.accept()

    sendSock, sendAddr = s.accept()

    print("Accepted")

    lock = mp.Lock()
    q = mp.Queue(40)

    jobs = []

    for i in range(3):
        jobs.append(mp.Process(target=processWorker, args=(c, lock, q)))

    jobs.append(mp.Process(target=inferAndSend, args=(q, sendSock,)))

    mp.Process(target=chart_data, ).start()

    for i in jobs:
        i.start()

    jobs[-1].join()
