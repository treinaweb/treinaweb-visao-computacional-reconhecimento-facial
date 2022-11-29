from deepface import DeepFace

analise = DeepFace.analyze(
    img_path = "img/Weverton.jpg",
    actions = ["age", "gender", "emotion"]
)

print(analise)