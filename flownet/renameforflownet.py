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
	for i in range(debut,fin+1) :
		if (i==debut):
			if(i<10):
				os.rename(f'{dossier}/frame00{i}.png',f'{dossier}/paire00{i}_1.png')
			elif(i<100):
				os.rename(f'{dossier}/frame0{i}.png',f'{dossier}/paire0{i}_1.png')
			else:
				os.rename(f'{dossier}/frame{i}.png',f'{dossier}/paire{i}_1.png')
		elif(i==fin) :
			if(i<10):
				os.rename(f'{dossier}/frame00{i}.png',f'{dossier}/paire00{i-1}_2.png')
			elif(i==10):
				os.rename(f'{dossier}/frame0{i}.png',f'{dossier}/paire00{i-1}_2.png')
			elif(i<100):
				os.rename(f'{dossier}/frame0{i}.png',f'{dossier}/paire0{i-1}_2.png')
			elif(i==100):
				os.rename(f'{dossier}/frame{i}.png',f'{dossier}/paire0{i-1}_2.png')
			else :
				os.rename(f'{dossier}/frame{i}.png',f'{dossier}/paire{i-1}_2.png')
		else : 
			if(i<10):
			shutil.copy(f'{dossier}/frame00{i}.png',f'{dossier}/paire00{i}_1.png')
				os.rename(f'{dossier}/frame00{i}.png',f'{dossier}/paire00{i-1}_2.png')
			elif(i==10):
				shutil.copy(f'{dossier}/frame0{i}.png',f'{dossier}/paire0{i}_1.png')
				os.rename(f'{dossier}/frame0{i}.png',f'{dossier}/paire00{i-1}_2.png')
			if(i<100):
				shutil.copy(f'{dossier}/frame0{i}.png',f'{dossier}/paire0{i}_1.png')
				os.rename(f'{dossier}/frame0{i}.png',f'{dossier}/paire0{i-1}_2.png')
			elif(i==100):
				shutil.copy(f'{dossier}/frame{i}.png',f'{dossier}/paire{i}_1.png')
				os.rename(f'{dossier}/frame{i}.png',f'{dossier}/paire0{i-1}_2.png')
			else :
				shutil.copy(f'{dossier}/frame{i}.png',f'{dossier}/paire{i}_1.png')
				os.rename(f'{dossier}/frame{i}.png',f'{dossier}/paire{i-1}_2.png')

	return 0


if __name__ == "__main__":
    main()
