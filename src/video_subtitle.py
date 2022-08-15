#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:26:51 2019

@author: onee
"""
import os
import cv2
import shutil
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
        if '/' in frames[i]:
            tmp = frames[i].split('/')
            frame_times.append(tmp[0].replace('pkt_pts_time=',''))
            frame_types.append(tmp[1])
    return frame_times, frame_types

def save_i_keyframes(video_fn,path):
    frame_times, frame_types = get_frame_types(video_fn)
    
    i_frames = []
    i_frames_times = []
    for i in range(len(frame_times)):
        if frame_types[i] == "I":
            i_frames.append(i)
            i_frames_times.append(frame_times[i])
            
    if i_frames:
        minsize = 20
        threshold = [ 0.6, 0.7, 0.7 ]  #three steps's threshold
        factor = 0.709 #scale factor
#        model = '../model/'
        image_size = 160
        margin = 44
        gpu_memory_fraction = 1.0
        
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
                
        cap = cv2.VideoCapture(video_fn)
        
        face_images = []
        file_list = []
#        video_name = []
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
#            cv2.imwrite(path+str(frame_no) +".jpg", frame)
#            outname = 'key'+str(frame_no)+'.jpg'
#            frameNum.append(frame_no)
#            cv2.imwrite(path+outname, frame)
#            f.write(str(frame_no)+str(" "))
#            img = cv2.imread(path+outname)
#            M = np.ones(frame.shape, dtype = "uint8") * 50 # 이미지 픽셀만큼 공간만들고, 100으로
#            frame = cv2.add(frame, M)
#            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            

            img_size = np.asarray(frame.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            person_num = 0
            
            for i in range(len(bounding_boxes)):
                if len(bounding_boxes) >= 1:
                    det = np.squeeze(bounding_boxes[i,0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-margin/2, 0)
                    bb[1] = np.maximum(det[1]-margin/2, 0)
                    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                    cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]
                    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    face_images.append(prewhitened)
                    file_list.append(str(frame_no) + "_"+str(person_num)+".jpg")
                    cv2.imwrite(path+str(frame_no) + "_"+str(person_num)+".jpg", cropped)
#                    video_name.append(str(frame_no) + "_"+str(person_num))
                    person_num = person_num + 1
            ###
#        f.write("\n")
#        f.close()
        faces = np.stack(face_images)
        cap.release()
          
    else:
        print ('No I-frames in '+video_fn)
        
    emb = make_emb(faces)
    
    return emb, file_list, i_frames, i_frames_times


def make_emb(face_img):
    model = '../20180402-114759/'

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


from pydub import AudioSegment 
from collections import Counter

def mp4_to_wav(video_path, filename):
    if os.path.isdir(video_path + "audio/"):
        shutil.rmtree(video_path + "audio/")
    os.makedirs(video_path + "audio/")
        
    #동영상 파일에서 오디오 추출
    print("음성추출시작")
    # 1. 영상 이름
    command = "ffmpeg -y -i "+ video_path + filename +" -ab 160k -ac 2 -ar 44100 -vn "+ video_path + "audio/" + filename + ".wav"
    subprocess.call(command, shell=True)
    print("음성추출완료")

def audio_split(video_path, filename):    
    f_name = filename.replace(".mp4","")
    if os.path.isdir(video_path + "audio/" + f_name):
        shutil.rmtree(video_path + "audio/" + f_name)
    os.makedirs(video_path + "audio/" + f_name)
    
    #자막 불러오기
    script = open(video_path + "subtitle/" + f_name + ".srt", 'r', encoding='UTF-8')
    
    lines = script.readlines()
        
    #오디오 파일 불러오기
    audio = AudioSegment.from_file(video_path + "audio/" + filename + ".wav", "wav")
    
    num = 0
    chunk_num = 0
    for i in lines:
        if (num % 4 == 1):
            tmp = i
            tmp = tmp.replace(" --> ", ":")
            tmp = tmp.replace(",", "")
            
            timestamp = tmp.split(':')
            #print(timestamp)
            
            #시작 ~ 끝 시간
            start_time = int(timestamp[0])*3600000 + int(timestamp[1])*60000 + int(timestamp[2])
            end_time = int(timestamp[3])*3600000 + int(timestamp[4])*60000 + int(timestamp[5])
            
            #print(start_time)
            
        if ( num % 4 == 2):
            s_tmp = i
            if ( s_tmp != "[음악]\n" and s_tmp != "[Music]\n" and s_tmp != "\n"):
                #오디오 추출
                chunk_data = audio[start_time:end_time]
                print(str(int(timestamp[0])*3600+int(timestamp[1])*60+int(timestamp[2])/1000))
                chunk_name = str(int(timestamp[0])*3600+int(timestamp[1])*60+int(timestamp[2])/1000)+"_"+str(int(timestamp[3])*3600+int(timestamp[4])*60+int(timestamp[5])/1000)+".wav"
                #폴더 직접 따로 생성해야함
                chunk_data.export(video_path + "audio/" + f_name + "/" + chunk_name, format="wav")
                chunk_num = chunk_num + 1
        num = num + 1
        
def readWavFile(video_path, filename, frame, sec, frame_cluster):
    f_name = filename.replace(".mp4","")
    voice = []
    fold_name = []
    for i in os.listdir(video_path + "audio/" + f_name +"/voice/"):
        if i != ".DS_Store":
            fold_name.append(i)
            char = []
            for j in os.listdir(video_path + "audio/" + f_name +"/voice/"+str(i)+'/'):
                if j != ".DS_Store":
                    char_tmp = []
                    temp = os.path.splitext(j)[0]
                    start = temp.split('_')[0]
                    end = temp.split('_')[1]
                    char_tmp.append(float(start))
                    char_tmp.append(float(end))
                    char.append(char_tmp)
            voice.append(char)
        
    fold_count = 0
    voice_name = {}
    for k in voice: # 사람수만큼
        temp=[]
        for m in k: # [시작,끝] pair만큼
            for i in range(len(frame_cluster)):
                for j in frame_cluster[i]: # j가 tmp 그 자체
                    idx = frame.index(int(j))
                    if float(sec[idx])>=m[0] and float(sec[idx])<m[1]:
                        temp.append(i)
                        
        c = Counter(temp)
        mode = c.most_common(1)
        if len(mode) > 0:
            c = mode[0][0]
            voice_name[str(fold_name[fold_count])] = c
#            if str(fold_name[fold_count]) in voice_name.keys():
#                print("있"+str(fold_name[fold_count]))
#                voice_name[str(fold_name[fold_count])].append(c)
#            else:
#                print("없"+str(fold_name[fold_count]))
#                voice_name[str(fold_name[fold_count])] = []
#                voice_name[str(fold_name[fold_count])].append(c)
        fold_count = fold_count+1
        
        makeSubtitle(video_path, f_name, voice_name)
        
def makeSubtitle(video_path, f_name, voice_name):
    #자막 불러오기
    script = open(video_path + "subtitle/" + f_name + ".srt", 'r', encoding='UTF-8')
    r_script = open(video_path + "subtitle/result_" + f_name + ".txt", 'w')
    
    num = 0
    lines = script.readlines()
    for i in lines:
        if (num % 4 == 1):
            tmp = i
            tmp = tmp.replace(" --> ", ":")
            tmp = tmp.replace(",", "")
            
            timestamp = tmp.split(':')
            
        if ( num % 4 == 2):
            if ( i != "[음악]\n" and i != "[Music]\n" and i != "\n"):
                chunk_name = str(int(timestamp[0])*3600+int(timestamp[1])*60+int(timestamp[2])/1000)+"_"+str(int(timestamp[3])*3600+int(timestamp[4])*60+int(timestamp[5])/1000)+".wav"
                
                flag = 0
                dir_n = ""
                for d in os.listdir(video_path + "audio/" + f_name + "/voice"):
                    if d != ".DS_Store":
                        if chunk_name in os.listdir(video_path + "audio/" + f_name + "/voice/"+d) and d in voice_name.keys():
                            flag = 1
                            dir_n = "<"+str(voice_name[d])+">\n"

                if flag == 1:
                    r_script.write(dir_n+str(i)+"\n\n")
                else:
                    r_script.write("<unknown>\n"+str(i)+"\n\n")                     
                    
        num = num + 1
        
    script.close()
    r_script.close()
    

from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    video_path = '../video/'
    filename = '3017.mp4'
    path = '../video/frame/'

    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    os.makedirs(path+'character')

    emb, file_list, i_frames, i_frames_times = save_i_keyframes(video_path+filename,path)


    db = DBSCAN(eps=0.69, min_samples=3).fit(emb)
    cluster = db.labels_


    f = 0
    for i in cluster:
        if not (os.path.isdir(path+'character/'+str(i))):
            os.makedirs(path+'character/'+str(i))

        shutil.copy(path+file_list[f], path+'character/'+str(i)+'/'+file_list[f])
#        shutil.move(path+file_list[f], path+'character/'+str(i)+'/'+file_list[f])
        f += 1

    set_cluster = list(set(cluster))
    set_cluster.remove(-1)
    frame_cluster = []
    for i in set_cluster:
#        print("Character "+str(i))
        tmp = []
        for j in np.where(i==cluster)[0]:
            tmp_frame = file_list[j].split('_')[0]
            tmp.append(tmp_frame)
        frame_cluster.append(tmp)
#            print(str(time_list[j][0])+' ~ '+str(time_list[j][1]))

#    makeTimeline(filename,last[0],float(last[1]),frame_cluster) # 타임라인 생성
        
#    mp4_to_wav(video_path, filename)
#    audio_split(video_path, filename)
    
#    음성 클러스터링 완료 후
    readWavFile(video_path, filename, i_frames, i_frames_times, frame_cluster)
