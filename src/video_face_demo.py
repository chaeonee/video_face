#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:53:52 2019

@author: onee
"""

import os
import cv2
import subprocess

from scipy import misc
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face

def get_frame_types(video_fn):#,pkt_pts_time
    command = 'ffprobe -v error -show_entries frame=pict_type,pkt_pts_time -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frames = out.replace('\npict_type=','/').split()

    frame_types = []
    frame_times = []
    for i in range(len(frames)):
        if frames[i].find('/') > -1:
            tmp = frames[i].split('/')
            frame_times.append(tmp[0].replace('pkt_pts_time=',''))
            frame_types.append(tmp[1])
    return frame_times, frame_types

def make_i_keyframes_emb(video_fn,path):
    frame_times, frame_types = get_frame_types(video_fn)
#    i_frames = [x[0] for x in frame_types if x[1]=='I']
    i_frames = []
    i_frames_times = []
    for i in range(len(frame_times)):
        if frame_types[i] == "I":
            i_frames.append(i)
            i_frames_times.append(frame_times[i])

    if i_frames:
        frame_size = 160
        margin = 44
        gpu_memory_fraction = 1.0

        minsize = 20 # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor

        print('Creating networks and loading parameters')

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        basename = os.path.splitext(os.path.basename(video_fn))[0]
        cap = cv2.VideoCapture(video_fn)

        face_list = []
        file_list = []
        time_list = []
        t = -1
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
#            cv2.imwrite('../video/'+str(t)+'.jpg', frame)
#            if ret == False:
#                continue

            t += 1

            bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)

            if len(bounding_boxes) < 1:
#                image_paths.remove(image)
                print("can't detect face, remove frame_", frame_no)
                continue

            f_size = np.asarray(frame.shape)[0:2]
            if not(os.path.isdir(path+str(frame_no))):
                os.makedirs(path+str(frame_no))


            for i in range(len(bounding_boxes)):
                det = np.squeeze(bounding_boxes[i,0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, f_size[1])
                bb[3] = np.minimum(det[3]+margin/2, f_size[0])
                cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]
                aligned = misc.imresize(cropped, (frame_size, frame_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                face_list.append(prewhitened)

                outname = path+str(frame_no)+'/'+basename+'_i_frame_'+str(frame_no)+'_face'+str(i)+'.jpg'
                file_list.append(outname)
                cv2.imwrite(outname, cropped)

                print ('Saved: '+outname)

                if t < len(i_frames_times)-1:
                    time_list.append([i_frames_times[t],i_frames_times[t+1]])
                else:
                    time_list.append([i_frames_times[t],0])
#            outname = path+basename+'_i_frame_'+str(frame_no)+'.jpg'
#            cv2.imwrite(outname, frame)
        faces = np.stack(face_list)
        cap.release()
    else:
        print ('No I-frames in '+video_fn)

    emb = make_emb(faces)

    return emb, file_list, time_list

def make_emb(face_img):
    model = '../model/'

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: face_img, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)

    return emb


import shutil

from sklearn.cluster import DBSCAN
#from sklearn.cluster import KMeans

if __name__ == '__main__':
    filename = '../video/vtest4.mp4'
    path = '../video/frame/'

    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    os.makedirs(path+'result')

    emb, file_list, time_list = make_i_keyframes_emb(filename,path)


    db = DBSCAN(eps=0.55, min_samples=4).fit(emb)
    cluster = db.labels_

#    km = KMeans(n_clusters=3, random_state=0).fit(emb)
#    cluster = km.labels_

    f = 0
    for i in cluster:
        if not(os.path.isdir(path+'result/'+str(i))):
            os.makedirs(path+'result/'+str(i))

        shutil.copy(file_list[f], path+'result/'+str(i)+'/'+str(f)+'.jpg')
        f += 1

    set_cluster = list(set(cluster))
    set_cluster.remove(-1)
    for i in set_cluster:
        print("Character "+str(i))
        for j in np.where(i==cluster)[0]:
            print(str(time_list[j][0])+' ~ '+str(time_list[j][1]))


#def video2frame(filename, path):
#    vidcap = cv2.VideoCapture(filename)

#    count = 0
#    while True:
#      success,image = vidcap.read()
#      if not success:
#          break
#      print ('Read a new frame: ', success)

#      faceCascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
#      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#      faces = faceCascade.detectMultiScale(
#              gray,
#              scaleFactor=1.3,
#              minNeighbors=5,
#              minSize=(100, 100)
#              )

#      print("Face Count : {0}".format(len(faces)))
#
#      if len(faces) != 0:
#          fname = "{}/".format("{0:05d}".format(count))
#          if not(os.path.isdir(path+fname)):
#              os.makedirs(path+fname)
##              cv2.imwrite(path + fname + "{}.jpg".format("{0:05d}".format(count)), image)

#      f_count = 0
#      for (x, y, w, h) in faces:
#          cv2.imwrite(path + fname + "face{}.jpg".format(f_count), image[y:y+h, x:x+w])
#          f_count += 1

#      faceCascade = cv2.CascadeClassifier('../haarcascades/lbpcascade_profileface.xml')

#      fname = "{}.jpg".format("{0:05d}".format(count))
#      cv2.imwrite(path + fname, image) # save frame as JPEG file
#      count += 1
#    print("{} images are extracted in {}.". format(count, path))

#import math
#def get_dist(p1,p2):
#    dist = 0
#    for i in range(0,len(p1)):
#        dist = dist + (p1[i]-p2[i])**2
##    dist = dist**4
#    dist = math.sqrt(dist)
#    return dist
#
#dist_list = []
#for i in range(len(emb)):
#    dist = []
#    for j in range(len(emb)):
#        dist.append(get_dist(emb[i],emb[j]))
#    dist_list.append(dist)


##재생할 파일의 넓이 얻기
#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
##재생할 파일의 높이 얻기
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
##재생할 파일의 프레임 레이트 얻기
#fps = cap.get(cv2.CAP_PROP_FPS)
#
#print('width {0}, height {1}, fps {2}'.format(width, height, fps))
