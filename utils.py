import os
import random
import shutil

def create_valid_data_from_train(train_images = "train/images", train_labels = "train/labels", num_images = len(os.listdir("train/images")), split_ratio=0.8):
    if not os.path.exists("valid"):
        os.makedirs("valid")
        os.makedirs("valid/images")
        os.makedirs("valid/labels")
    else:
        print("valid folder already exists")
        print("Exiting...")
        return

    train_image_files = os.listdir(train_images)
    train_label_files = os.listdir(train_labels)

    valid_size = int(num_images * (1 - split_ratio))
    valid_indexes = random.sample(range(num_images), valid_size)
    
    for index in valid_indexes:
        shutil.move(
            os.path.join(train_images, train_image_files[index]), 
            os.path.join("valid/images", train_image_files[index])
        )
        shutil.move(
            os.path.join(train_labels, train_label_files[index]), 
            os.path.join("valid/labels", train_label_files[index])
        )
    
    print("valididation data created successfully")
    print("Old train data size: ", num_images)
    print("New train data size: ", len(os.listdir("train/images")))
    print("valididation data size: ", len(os.listdir("valid/images")))
    return

def reset_valid_data():
    if os.path.exists("valid"):
        valid_image_files = os.listdir("valid/images")
        valid_label_files = os.listdir("valid/labels")
        for index in range(len(os.listdir("valid/images"))):
            shutil.move(
                os.path.join("valid/images", valid_image_files[index]), 
                os.path.join("train/images", valid_image_files[index])
            )
            shutil.move(
                os.path.join("valid/labels", valid_label_files[index]), 
                os.path.join("train/labels", valid_label_files[index])
            )
        os.rmdir("valid/images")
        os.rmdir("valid/labels")
        os.rmdir("valid")
        print("valididation data reset successfully")


if __name__ == "__main__":
    create_valid_data_from_train()
    # reset_valid_data()
    
