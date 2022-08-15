from collections import Counter
import os
filename = "vvv.mp4"
def readFile():
    f = open("../data/"+filename+"/"+filename+".txt", 'r')

    voice = []
    fold_name = []
    for i in os.listdir('../voice/'):
        fold_name.append(i)
        char = []
        for j in os.listdir('../voice/'+str(i)+'/'):
            char_tmp = []
            temp = os.path.splitext(j)[0]
            start = temp.split('_')[0];
            end = temp.split('_')[1];
            char_tmp.append(float(start))
            char_tmp.append(float(end))
            char.append(char_tmp)
        voice.append(char)


    lines = f.readlines()
    frame = lines[1].split(' ')
    sec = lines[2].split(' ')
    fold_count = 0;
    for k in voice:
        temp=[]
        for m in k:
            for i in range(5,len(lines)):
                char_frame = lines[i].split()
                for j in range(1, len(char_frame)):
                    tmp = char_frame[j].split('_')[0]
                    for l in range(len(frame)-1):
                        if(frame[l]==tmp):
                            if(float(sec[l])>=m[0] and float(sec[l])<m[1]):
                                temp.append(char_frame[0])

        c = Counter(temp)
        mode = c.most_common(1)
        c = mode[0][0]
        print(fold_name[fold_count] +" "+ c)
        fold_count = fold_count+1
    f.close()


if __name__ == '__main__':
    readFile()
