import os  
import sys
import shutil


def main():
	dossier=sys.argv[1:][0]
	if not os.path.exists(dossier):
		print("le dossier n'existe pas") 
		return(0)
	debut=int(sys.argv[2:][0])
	fin=int(sys.argv[3:][0])
	nb_frames=fin-debut
	for i in range(0,nb_frames+1) :
		if(i<=9):
			if(debut+i<10):
				os.rename(f'paire00{i+debut}_flow.png',f'flow00{i}.png')
			elif(debut+i<100):
				os.rename(f'paire0{i+debut}_flow.png',f'flow00{i}.png')
			else:
				os.rename(f'paire{i+debut}_flow.png',f'flow00{i}.png')
		else:
			elif(debut+i<100):
				os.rename(f'paire0{i+debut}_flow.png',f'flow0{i}.png')
			elif(i<100):
				os.rename(f'paire{i+debut}_flow.png',f'flow0{i}.png')
			else:
				os.rename(f'paire{i+debut}_flow.png',f'flow{i}.png')
	return 0

if __name__ == "__main__":
	main()
