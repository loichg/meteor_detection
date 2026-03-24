import numpy as np
import cv2
import matplotlib.image as mpimg
import os
import random
import imutils
from path import Path
import argparse

def coller_img1_sur_bg(img1,bg,pos):
    h, w, couleurs = img1.shape
    in_bounds=True
    for i in range(w):
        for j in range(h):
            if not np.array_equal(img1[j, i], [0, 0, 0]):
                if(0<j+pos[1]-int(h/2)<436 and 0<i+pos[0]+int(w/2)<1024):
                    bg[j+pos[1]-int(h/2),i+pos[0]+int(w/2)]=img1[j,i]
                else :
                    in_bounds=False

    return bg,in_bounds


def coller_img1_sur_bg2(img1,bg,pos,flow,deplacement_x,deplacement_y,deplacement_bg_x,deplacement_bg_y):
    h, w, couleurs = img1.shape
    in_bounds=True
    for i in range(w):
        for j in range(h):
            if not np.array_equal(img1[j, i], [0, 0, 0]):
                if(0<j+pos[1]-deplacement_bg_y+deplacement_y-int(h/2)<436 and 0<i+pos[0]-deplacement_bg_x+deplacement_x+int(w/2)<1024):
                    bg[j+pos[1]-deplacement_bg_y+deplacement_y-int(h/2),i+pos[0]-deplacement_bg_x+deplacement_x+int(w/2)]=img1[j,i]
                    flow[j+pos[1]-int(h/2)][i+pos[0]+int(w/2)][0]+=deplacement_x
                    flow[j+pos[1]-int(h/2)][i+pos[0]+int(w/2)][1]+=deplacement_y
                else : 
                    in_bounds=False

    return bg,flow,in_bounds



def translation(im_bg,deplacement_x,deplacement_y):
        w=1024
        h=436
        frame1=np.zeros((h,w,3))
        frame2=np.zeros((h,w,3))
        for i in range (w):
            for j in range(h):
                frame1[j,i]=im_bg[j+int((536-h)/2),i+int((1124-w)/2)]
                frame2[j,i]=im_bg[j+int((536-h)/2)+deplacement_y,i+int((1124-w)/2)+deplacement_x]
        return frame1,frame2





def write_flo(filename, flow):
    """
    Sauvegarde un flot optique dans un fichier .flo.

    :param filename: Nom du fichier de sortie
    :param flow: Tableau numpy de dimensions (h, w, 2) contenant le flot optique
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("Le flot optique doit avoir la forme (h, w, 2)")

    h, w = flow.shape[:2]
    
    with open(filename, 'wb') as f:
        # Écriture du header
        f.write(np.array([202021.25], dtype=np.float32).tobytes())  # Magic number
        f.write(np.array([w, h], dtype=np.int32).tobytes())         # Largeur et hauteur

        # Écriture des données du flot optique
        f.write(flow.astype(np.float32).tobytes())




def genere_data(dataset,nb_pair_frames,ranger):
    if (ranger):
        path_clean=Path(dataset)/"clean"
        path_flow=Path(dataset)/"flow"
        if (os.path.exists(path_clean)==False):
            path_clean.makedirs_p()
        if (os.path.exists(path_flow)==False):
            path_flow.makedirs_p()

    for k in range (nb_pair_frames):
        w=1024
        h=436
        padding_x=40
        padding_y=40
        pos_x=random.randint(padding_x,w-padding_x)
        pos_y=random.randint(padding_y,h-padding_y)
        teta=random.randint(0,360)
        deplacement=random.randint(4,10)
        deplacement_x=int(np.cos(teta*np.pi/180)*deplacement)
        deplacement_y=int(np.sin(teta*np.pi/180)*deplacement)
        met=random.randint(1,51)
        bg=random.randint(0,622)

        im_met=cv2.imread(f'motif/met{met}.png')
        im_bg=cv2.imread(f'bg/output{bg}.png')

        random_bg=random.randint(1,10)
        if (random_bg<=4):
            deplacement_bg=0
        else:
            deplacement_bg=2
        
        teta_bg=random.randint(0,360)

        deplacement_bg_x=int(np.cos(teta_bg*np.pi/180)*deplacement_bg)
        deplacement_bg_y=int(np.sin(teta_bg*np.pi/180)*deplacement_bg)

        flow=np.zeros((h,w,2))
        for i in range(w):
            for j in range(h):
                flow[j][i][0]=-deplacement_bg_x
                flow[j][i][1]=-deplacement_bg_y

        (im_bg1,im_bg2)=translation(im_bg,deplacement_bg_x,deplacement_bg_y)
        rotated_met = imutils.rotate_bound(im_met, teta)

        in_bounds2=True
        (output1,in_bounds1)=coller_img1_sur_bg(rotated_met,im_bg1,[pos_x,pos_y])
        if (in_bounds1):
            (output2,flow,in_bounds2)=coller_img1_sur_bg2(rotated_met,im_bg2,[pos_x,pos_y],flow,deplacement_x,deplacement_y,deplacement_bg_x,deplacement_bg_y)

        if (in_bounds1 and in_bounds2) :
            if (ranger):
                path_clean_pair=path_clean/f'pair{k}'
                path_clean_pair.makedirs_p()
                path_flow_pair=path_flow/f'pair{k}'
                path_flow_pair.makedirs_p()
                cv2.imwrite(path_clean_pair/'frame_0001.png',output1)
                cv2.imwrite(path_clean_pair/'frame_0002.png',output2)
                write_flo(path_flow_pair/'frame_0001.flo', flow)
            else :
                cv2.imwrite(f'pair{k}_frame1.png',output1)
                cv2.imwrite(f'pair{k}_frame2.png',output2)
                write_flo(f'flo_pair{k}.flo', flow)

        #chargement :
        if (int(k*100/nb_pair_frames)%10==0):
            print(int(k*100/nb_pair_frames),"%")

    return 0





parser = argparse.ArgumentParser(
    description="PyTorch FlowNet inference on a folder of img pairs",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--nb_paires",
    default=10,
    type=int,
    help="nombre de paires à générer",
)



def main():
    global args, save_path
    args = parser.parse_args()
    print("génération de : ",args.nb_paires," paires")
    genere_data("dataset_syn",args.nb_paires,1)
    return(0)



if __name__ == "__main__":
    main()