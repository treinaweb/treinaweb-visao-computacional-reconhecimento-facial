from deepface import DeepFace

verificacao = DeepFace.verify(img1_path="img/Veiga.jpg",
                              img2_path="img/GustavoGomez.jpg")
verificacao2 = DeepFace.verify(img1_path="img/Weverton.jpg",
                              img2_path="img/ZeRafael.jpg")
print(verificacao2)

