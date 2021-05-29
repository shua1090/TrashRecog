import multiprocessing as mp
import os
import pickle
import socket
import threading
import time

import cv2
import lz4.block as compressingEngine

from Light import light


def display(sock: socket, lock):

    l = light()
    l.setupFlash()

    while True:
        print("received: ")

        lock.acquire()
        b = pickle.loads(sock.recv(1024))
        lock.release()

        if b == "END":
            break

        l.lightUp(b)


def frame_getter(cam, queue, lock):
    try:
        while True:
            lock.acquire()
            ret, frame = cam.read()
            lock.release()
            queue.put(frame, True)
    except:
        cam.release()


def frame_worker(pipe_queue, sock1):
    cam = cv2.VideoCapture(0, )

    lock1 = mp.Lock()
    threading.Thread(target=display, args=(sock1, lock1)).start()

    lock2 = mp.Lock()

    threading.Thread(target=frame_getter, args=(cam, pipe_queue, lock2)).start()
    t = threading.Thread(target=frame_getter, args=(cam, pipe_queue, lock2))
    t.start()
    t.join()


def thread_worker(q, pass_lock, conn):
    print("-" * 10 + f"Worker {os.getpid()} started" + "-" * 10)

    while True:
        z = q.get(True)

        height, width, _ = z.shape
        dim = 224

        resized = cv2.resize(z, (dim, dim))

        a = compressingEngine.compress(pickle.dumps(resized, protocol=3))  # , 2)
        pass_lock.acquire()
        conn.send(pickle.dumps(len(a), protocol=3))
        conn.send(a)
        pass_lock.release()

        print(f"{os.getpid()} sent")
        time.sleep(0.010)


def worker(q, lock, conn):
    threading.Thread(target=thread_worker, args=(q, lock, conn)).start()
    t = threading.Thread(target=thread_worker, args=(q, lock, conn))
    t.start()
    t.join()


if __name__ == "__main__":
    jobs = []
    parent_conn, child_conn = mp.Pipe()
    lock = mp.Lock()

    sockA = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 5000
    address = ('192.168.86.110', port)
    sockA.connect(address)

    sockB = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sockB.connect(address)

    queue = mp.Queue(40)
    jobs.append(mp.Process(target=frame_worker, args=(queue, sockB)))

    for i in range(3):
        work = mp.Process(target=worker, args=(queue, lock, sockA))
        jobs.append(work)

    for i in jobs:
        i.start()

    t1 = time.time()

    while jobs[1].is_alive():
        pass

    jobs[0].terminate()

    print(f"Total Time that has occurred: {time.time() - t1}")
