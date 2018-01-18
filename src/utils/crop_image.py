import os
import cv2
import multiprocessing

NUM_WORKERS = 4

DATA_DIR = "data/raw/training/big_images"
OUT_DIR = "data/raw/training/00100"


def crop_and_save(iterator):
    no, file_name = iterator
    try:
        print("something")
        img = cv2.imread(file_name)
        size = min(img.shape[:2])
        print(size)
        stride = 200
        for i in range(int(size / stride)):
            for j in range(int(size / stride)):
                path = OUT_DIR + "/stride" + str(50) + \
                    str(no) + "_" + str(i) + "_" + str(j) + ".jpg"
                print(path)
                cv2.imwrite(
                    path, img[i * stride:i * stride + stride,
                              j * stride:j * stride + stride])
    except Exception as e:
        print(e)


def main():
    file_names = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(
        ".ppm") or f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    print("There are totally ", len(file_names), " files")

    # For each label, load it's images and add them to the images list.
    # And add the label number (i.e. directory name) to the labels list.
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        print("smthing")
        results = pool.map_async(crop_and_save, enumerate(file_names))
        results.wait()

    # for no, f in enumerate(file_names):
    #     img = cv2.imread(f)
    #     size = min(img.shape[:2])
    #     print(size)
    #     stride = 50
    #     for i in range(int(size / stride)):
    #         for j in range(int(size / stride)):
    #             path = out_dir + "/stride50_" + \
    #                 str(no) + "_" + str(i) + "_" + str(j) + ".jpg"
    #             print(path)
    #             cv2.imwrite(
    #                 path, img[i * stride:i * stride + stride,
    #                           j * stride:j * stride + stride])


if __name__ == '__main__':
    main()
