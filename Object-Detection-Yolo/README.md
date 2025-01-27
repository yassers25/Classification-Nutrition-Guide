

config.yaml : rédirige les chemins (dataset , données d'entrainement ,données de validation , les noms des classes)

main.py : entrainement encore le modèle (j'ai arreter en train13 ,  en globe 32 epochs mais j'ai trouver le modele performant en epoch 13)

predict_gifs, video , images .py

assemble_resultats.py
combined_results.csv

affichage_results.py : affiche les resultats dans output , et ajoute dans train13 dossier smito vala tous  les visualizations que donne Yolo par defaut( Matrice de confusion) 

--image-dir : argument (chemin d'images que tu veux tester  )
--input-dir : argument (chemin des gifs que tu veux tester   )
--output-dir : argument (chemin du dossier ou tu  veux poser resultats des tests)
### commandes:
1. pip install ultralytics cv2 tqdm pycocotools

2. python predict_images.py --image-dir /path/to/images --input-dir /path/to/folders_of_resluts

ex : python predict_images.py --image-dir "C:\Users\HP\Downloads\Object-Detection-Yolo\Object-Detection-Yolo\Test\test" 
--output-dir "C:\Users\HP\Downloads\Object-Detection-Yolo\Object-Detection-Yolo\Test\results of test" 

3. python predict_gifs.py --input-dir /path/to/images --input-dir /path/to/folders_of_resluts
ex : python predict_gifs.py --input-dir "C:\Users\HP\Downloads\Test\test\gifs" --output-gif "C:\Users\HP\Downloads\Test\results of test\gifs"


5. python assemble_results.py

5. python afficher_results.py


