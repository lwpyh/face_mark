# coding: utf8
import os
import pickle
import face_recognition as fr

images_root = "/home/tuo/jhu/SCUT-FBP5500_v2/SCUT-FBP5500_v2/Images"
labels_file = "/home/tuo/jhu/SCUT-FBP5500_v2/SCUT-FBP5500_v2/train_test_files/All_labels.txt"

face_list = []
with open(labels_file, encoding = "utf8") as fi:
    for line in fi:
        line = line.strip()
        filename, score = line.split(" ")
        full_path = os.path.join(images_root, filename)
        if not os.path.exists(full_path):
            print("Error: image file not found: ", full_path)
            continue
        image = fr.load_image_file(full_path)
        assert image is not None
        encs = fr.face_encodings(image)
        if len(encs) != 1:
            print("%s has %d faces." % (filename, len(encs)))
            continue

        item = {
            'enc': encs[0],
            'score': float(score)
        }
        face_list.append(item)

print("Total faces: %d." % len(face_list))
with open("training.pkl", "wb") as fo:
    pickle.dump(face_list, fo)