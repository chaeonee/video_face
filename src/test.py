from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import ImageTk, Image

from scipy import misc, spatial
import tensorflow as tf
import cv2
import subprocess
import numpy as np
import os
import copy
import facenet
import align.detect_face
import shutil
from tkinter import Canvas
import tkinter as tk

filepath = '../video/'
filename = 'vvvv.mp4'

frameNum = []
width = 640
height = 3000
bpp = 3

img = np.zeros((height, width, bpp), np.uint8)
img_h = img.shape[0]
img_w = img.shape[1]
img_bpp = img.shape[2]
main_character = []

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
cyan = (255, 255, 0)
magenta = (255, 0, 255)
thickness = 3
thickness2 = -1

def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    command2 = 'ffprobe -v error -show_entries frame=pict_type,pkt_dts_time -of default=noprint_wrappers=1'.split()

    out = subprocess.check_output(command + [video_fn]).decode()
    out2= subprocess.check_output(command2 + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    frames = out2.replace('\npict_type=','/').split()
    print(frames)
    frame_times = []
    for i in range(len(frames)):
        if frames[i].find('/') <= -1:
            tmp = frames[i]
            frame_times.append(tmp.replace('pkt_dts_time=',''))

    return zip(range(len(frame_types)), frame_types), frame_times[len(frame_times)-1]

def save_i_keyframes(video_fn):
    frame_types, frame_times = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    minsize = 20
    threshold = [ 0.6, 0.7, 0.7 ]  #three steps's threshold
    factor = 0.709 #scale factor
    model = '../model/'
    image_size = 160
    margin = 44
    gpu_memory_fraction = 1.0

    os.mkdir("../data/"+filename)
    f = open("../data/"+filename+"/"+filename+".txt", 'w')
    f.write(str(frame_times))
    f.write('\n')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    if i_frames:
        cap = cv2.VideoCapture(video_fn)
        face_images = []
        video_name = []

        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = 'key'+str(frame_no)+'.jpg'
            frameNum.append(frame_no)
            cv2.imwrite("../keyframe/"+outname, frame)
            f.write(str(frame_no)+str(" "))
            img = cv2.imread("../keyframe/"+'key'+str(frame_no)+'.jpg')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            person_num = 0

            for i in range(len(bounding_boxes)):
                if len(bounding_boxes) >= 1:
                    det = np.squeeze(bounding_boxes[i,0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-margin/2, 0)
                    bb[1] = np.maximum(det[1]-margin/2, 0)
                    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    face_images.append(cropped)
                    cv2.imwrite("../keyframe/"+str(frame_no) + "_"+str(person_num)+".jpg", cropped)
                    video_name.append(str(frame_no) + "_"+str(person_num))
                    person_num = person_num + 1
            ###
        f.write("\n")
        for frame_no in i_frames:
            f.write(str(round((float(frame_times)/float(i_frames[len(i_frames)-1]))*float(frame_no),4))+" ")
        f.write("\n")
        f.close()
        cap.release()

        # Compare image
        image_path = '../main_character/'
        compare(face_images, model, image_path, image_size, margin, gpu_memory_fraction, video_name)

    else:
        print ('No I-frames in '+video_fn)




def tuple_add(x, y):
    return tuple(np.add(x, y))

def saveTimeline():
    f = open("../data/"+filename+"/"+filename+".txt", 'a')

    runningTime = frameNum[len(frameNum)-1]
    f.write(str(runningTime))
    f.write('\n')


    for i in os.listdir('../main_character/'):

        if i in main_character:
            f.write(str(i))
            f.write(" ")
            img_list = []
            path = '../main_character/'+str(i)+'/'

            for j in os.listdir(path):
                f.write(str(j))
                f.write(" ")
                img_list.append(str(j))

            img = cv2.imread('../main_character/'+str(i)+'/'+img_list[0])
            img2 = Image.fromarray(img)
            img2.save("../data/"+str(filename)+"/"+img_list[0])
            f.write('\n')

    f.close()


def makeTimeline():

    root = tk.Tk()

    #Create a canvas
    canvas = Canvas(root, width=1000, height=1000)
    canvas.pack()


    ############################################
    runningTime = frameNum[len(frameNum)-1]

    character_path1 = []
    character_path = '../main_character/'
    character_emerge = []
    character_count = 0
    count = 0;
    im1 = []
    image_files = []
    count2 = []
    t = 0
    file = open("../data/"+filename+"/"+filename+".txt", 'a')
    for i in os.listdir('../main_character/'):
        k = 0
        if not i == '.DS_Store':
            for f in os.listdir(os.path.join('../main_character/', i)):
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                     image_files.append(os.path.join('../main_character/', i, f))
                k = k+1
        count2.append(k)    #폴더 내 이미지 수(등장횟수)
        if count2[t]!=0 :
            character_count = character_count + 1   #사람수
        t = t+1
    character_count = len(image_files)/character_count
    file.write(str(character_count))
    file.write('\n')
    file.close()

    #cv2.rectangle(img, tuple_add((0,0), (10, 10)), tuple_add((img_w - 1, img_h - 1), (-10, -200)), blue, thickness)
    for a in os.listdir('../main_character/'):
        character_path1.append(os.path.join(character_path,a))
    for k in range(len(character_path1)):
        character_path = character_path1[k]+'/';
        character_emerge = []

        try:
            os.rmdir(character_path)

        except: #빈 폴더가 아니면
            for i in os.listdir(character_path):
                character_emerge.append(i)

            if(len(character_emerge)>=character_count):
                im = Image.open(character_path+character_emerge[0]).resize((100,100), Image.ANTIALIAS)
                im1.append(ImageTk.PhotoImage(im))
                canvas.create_image(0, count*(100), image=im1[len(im1)-1], anchor='nw')
                for i in range(0,len(character_emerge)):
                    for j in range(0,len(frameNum)-1):
                        if(str(frameNum[j])+".jpg"==str(character_emerge[i])):
                            check_start = frameNum[j]
                            check_end = frameNum[j+1]
                            ratio_start = int(float((width/runningTime)*check_start))
                            ratio_end = int(float((width/runningTime)*check_end))



                            canvas.create_rectangle(110+ratio_start,10+count*(100), 100+ratio_end,100+count*(100), fill = 'red')

                count = count + 1

    root.mainloop()


def compare(input_images, model, image_path, image_size, margin, gpu_memory_fraction, video_name):
    image_files = []

    for i in os.listdir(image_path):
        if not i == '.DS_Store':
            for f in os.listdir(os.path.join(image_path, i)):
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                     image_files.append(os.path.join(image_path, i, f))
    images = load_and_align_data(input_images, image_files, image_size, margin, gpu_memory_fraction)

    with tf.Graph().as_default():

        with tf.Session() as sess:  #그래프 실행하기 위해 tf.Session생성
                                    #세션에 아무런 파라미터도 넘기지 않았다는 것은 기본 로컬세션에서 수행됨을 의미

            # Load the model
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            # 학습할 예제
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)


            for i in range(len(input_images)):
                dist_list = []

                for j in range(len(input_images), len(emb)):
                    dist = spatial.distance.cosine(emb[i,:], emb[j,:])

                    dist_list.append(dist)

                if (len(dist_list) == 0 or min(dist_list) > 0.2):
                    os.mkdir(image_path + str(i))
                    pil_image = Image.fromarray(input_images[i])
                    pil_image.save(image_path+str(i)+"/"+str(video_name[i])+".jpg")
                    image_files=[]
                    for k in os.listdir(image_path):
                        if not k == '.DS_Store':
                            for f in os.listdir(os.path.join(image_path, k)):
                                ext = os.path.splitext(f)[1]
                                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                                    image_files.append(os.path.join(image_path, k, f))

                    images = load_and_align_data(input_images, image_files, image_size, margin, gpu_memory_fraction)

                                        # Run forward pass to calculate embeddings
                    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                    emb = sess.run(embeddings, feed_dict=feed_dict)


                elif (min(dist_list)<=0.2):
                    dist_index = dist_list.index(min(dist_list))
                    person_path = os.path.abspath(os.path.join(image_files[dist_index], '..'))
                    person_name = person_path.split('/')[-1]
                    pil_image = Image.fromarray(input_images[i])
                    pil_image.save(person_name+"/"+str(video_name[i])+".jpg")
                else :
                    print("Pass")





def load_and_align_data(input_images, image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths=copy.copy(image_paths)

    img_list = []

    for i in range(len(input_images)):
        aligned = misc.imresize(input_images[i], (image_size, image_size), interp='bilinear')

        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)

    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          os.remove(image)


        else:
            det = np.squeeze(bounding_boxes[0,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
    images = np.stack(img_list)

    return images


def load_and_align_data2(input_images, image_size, margin, gpu_memory_fraction):



    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    #tmp_image_paths=copy.copy(image_paths)

    img_list = []
    for i in range(len(input_images)):
        aligned = misc.imresize(input_images[i], (image_size, image_size), interp='bilinear')

        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)

    images = np.stack(img_list)

    return images


def findMedoid():
    #folder_num = len(next(os.walk('/tmp/'))[1])
    model = '../model/'


    #folder_num = len(next(os.walk('/medoid_test/'))[1])

    medoid_img = [] #중앙 이미지 저장
    folder_name =[] #각 캐릭터 저장되어있는 폴더 이름
    fold_path = '../main_character/'
    tmp_path = []
    tmp_foldpath = []

    med_min = 10000
    for i in os.listdir('../main_character/'):
        folder_name.append(os.path.join(fold_path, i))
    with tf.Graph().as_default():
        with tf.Session() as sess:
                        # Load the model
            facenet.load_model(model)
                        # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            for j in range(len(folder_name)):
                medoid_path = folder_name[j]+'/'
                image_files = []
                image_path = []
                for i in os.listdir(medoid_path):   #medoid찾을 path 즉, 각 폴더 경로에 있는 사진들
                    image_path.append(os.path.join(medoid_path, i))
                    img = cv2.imread(os.path.join(medoid_path, i))

                    image_files.append(img)

                if (len(image_files)>0):
                    images = load_and_align_data2(image_files, 160, 44, 1.0)

                    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                    emb = sess.run(embeddings, feed_dict=feed_dict)

                    med_num = 0
                    for k in range(len(image_files)):
                        dist_list = []
                        for j in range(len(image_files), len(emb)):
                            dist = spatial.distance.cosine(emb[k,:], emb[j,:])
                            dist_list.append(dist)
                        sum = 0 #
                        for l in dist_list:
                            sum = sum + l

                        if (sum <= med_min) :
                            med_min = sum
                            med_num = k
                    medoid_img.append(image_files[med_num])  #중앙 이미지 자체 저장
                    tmp_path.append(image_path[med_num])    #중앙 이미지의 path 저장
                    tmp_foldpath.append(medoid_path)    #중앙 이미지가 저장되어 있는 폴더의 path

    folder_merge(medoid_img, tmp_path, tmp_foldpath, model)




def folder_merge(medoid_img, tmp_path, tmp_foldpath, model):
    images = load_and_align_data2(medoid_img, 160, 44, 1.0)
    del_list = []
    del_img = []
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)

            for i in range(len(medoid_img)):
                #dist_list = []
                k = 0
                check = 0
                for j in range(i+1, len(medoid_img)):

                    for m in range(len(del_img)):
                        if (j== m):
                            check = 1
                    if (check == 0):
                        dist = spatial.distance.cosine(emb[i,:], emb[j,:])

                                    #dist_list.append(dist)

                        if (dist <= 0.4): #import shutil해야함
                            print(str(tmp_path[i])+str(tmp_path[j])+"는 동일인물");
                            for l in os.listdir(tmp_foldpath[j]):
                                shutil.copy(tmp_foldpath[j]+l, tmp_foldpath[i])
                            del_list.append(tmp_foldpath[j])
                            del_img.append(j)
                        else:
                            print(str(tmp_path[i])+str(tmp_path[j])+"는 다른인물");
                        k=k+1

    for i in range(len(del_list)):
        try:
            shutil.rmtree(del_list[i])
        except:
            continue

def find_main_character(image_path):
    character_count = 0
    image_files = []
    count = []
    t = 0
    for i in os.listdir(image_path):
        k = 0
        if not i == '.DS_Store':
            for f in os.listdir(os.path.join(image_path, i)):
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                     image_files.append(os.path.join(image_path, i, f))
                k = k+1
        count.append(k)
        if count[t]!=0 :
            character_count = character_count + 1
        t = t+1

    print("총 얼굴 수 : "+str(len(image_files)))
    character_count = len(image_files)/character_count
    print("주인공 판별 기준 : " + str(character_count) +"이상")
    for i in range(len(count)):
        if(count[i]>=character_count):
            print(str(i)+"번째 주인공 : 폴더 " + str(os.listdir(image_path)[i]))
            main_character.append(str(os.listdir(image_path)[i]))

def findRelationship(file):
    f = open("../data/"+file+"/"+file+".txt", 'r')
    lines = f.readlines()

    for i in range(3, len(lines)):
        for j in range(i+1, len(lines)):
            l1 = lines[i].split()
            l2 = lines[j].split()

            count = 0
            for k in range(len(l1)):
                for l in range(len(l2)):
                    if (l1[k] == l2[l]):
                        count = count + 1
                        print(l1[0]+"과"+l2[0]+"는 "+str(count)+"번 동시 등장")

    f.close()


if __name__ == '__main__':
    save_i_keyframes(filepath+filename)
    #findMedoid()
    find_main_character('../main_character/')
    makeTimeline()
    saveTimeline()
    #findRelationship(filename)
