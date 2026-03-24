
import cv2
import argparse
import os
import numpy as np
from path import Path

def gen_noyau_lissage(p):
    L=[]
    for k in range(2*p+1):
        L.append([1/((2*p+1)**2)] * (2*p+1)) 
    return L
    



def convo_loc2(i,j,p,flow_output,L):
    s1=0
    s2=0
    for k in range (2*p+1):
        for l in range (2*p+1):
            v1=flow_output[j+l-p][i+k-p][0]
            v2=flow_output[j+l-p][i+k-p][1]
            s1=s1+(v1*(L[l][k]))
            s2=s2+(v2*(L[l][k]))

    return (s1,s2)


def create_im_vect2(img1_file,name_img_vect,flow_output,L,facteur):
    # shutil.copy(img1_file,name_img_vect)
    img = cv2.imread(img1_file)
    w=len(flow_output[0])
    h=len(flow_output)
    img=cv2.resize(img, (w,h), interpolation = cv2.INTER_AREA)
    if img is None:
        print('impossible de charger les données du fichier')
    # img=cv2.arrowedLine(img, start_pt, end_pt, (0, 255, 0), 9)

    p=len(L)//2
    for i in range (p,w-p,p): 
        for j in range (p,h-p,p):
            # (start_pt,end_pt)=([0,0],[0,0])
            start_pt=[i,j]
            (e1,e2)=convo_loc2(i,j,p,flow_output,L)
            end_pt=(i+int(e1*facteur),j+int(e2*facteur))
            img=cv2.arrowedLine(img, start_pt, end_pt, (70, 255, 255), 1, cv2.LINE_AA, 0, 0.1)

    cv2.imwrite(name_img_vect, img)

    return 0



parser = argparse.ArgumentParser(
    description="PyTorch FlowNet inference on a folder of img pairs",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--arrow_size",
    default=0.5,
    type=float,
    help="size of the arrows for the vector images",
)

parser.add_argument(
    "--arrow_segmentation",
    default=9,
    type=int,
    help="space between the arrows for the vector images",
)

parser.add_argument(
    "--folder",
    "-f",
    metavar="DIR",
    default=None,
    help="path where the folder flow is",
)

parser.add_argument(
    "--begin_frame",
    "-b",
    default=1,
    type=int,
    help="int which matches with the number of the first frame (default:1)",
)

parser.add_argument(
    "--end_frame",
    "-e",
    default=500,
    type=int,
    help="int which matches with the number of the end frame (default:500)",
)




def main():
    global args, save_path
    args = parser.parse_args()
    L=gen_noyau_lissage(args.arrow_segmentation)
    facteur=args.arrow_size
    path_vectors=Path(args.folder) / "flow/vectors"
    path_vectors.makedirs_p()
    for k in range (args.begin_frame,args.end_frame):
        im_path=Path(args.folder)/ "paire{}_1.png".format(k)
        file_path=Path(args.folder)/"flow/"+"paire{}_flow.npy".format(k)
        file_output=Path(args.folder)/"flow/vectors/"+"paire{}_vect.png".format(k)
        if os.path.exists(file_path):
            flow_output=np.load(file_path)
            create_im_vect2(im_path,file_output,flow_output,L,facteur)
    

if __name__ == "__main__":
    main()
