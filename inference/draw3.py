import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

colors = [[41, 128, 185], [52, 152, 219], [26, 188, 156], [22, 160, 133], [39, 174, 96],
          [46, 204, 113], [241, 196, 15], [243, 156, 18], [230, 126, 34], [211, 84, 0]
          ]
color_len = len(colors)
# print(color_len)
font_size = 10
base_path_yolo = './hello'
base_path_eyolo = './hello_ua'
base_path_mmyolo = './hello_ro'

ori_list = []
arr1_list = []
arr2_list = []
arr3_list = []


def anchor2mask(anchor_image):
    mask_color_2 = np.array(colors, dtype=np.uint8)[anchor_image]
    return mask_color_2[..., ::-1]


def show_plts(original_list, array1_list, array2_list, array3_list):

    assert len(original_list) == len(array1_list) == len(array2_list) == len(array3_list)
    rows = len(original_list)
    fig, axs = plt.subplots(nrows=rows, ncols=4, figsize=(8, 12), subplot_kw={'xticks': [], 'yticks': []})

    for idx in range(len(original_list)):
        if idx == rows - 1:
            axs[idx, 0].set_xlabel('Original Image', fontdict={'size': font_size})
        if idx % 3 == 0:
            axs[idx, 0].set_ylabel('MME-YOLO', fontdict={'size': 8})
        elif idx % 3 == 1:
            axs[idx, 0].set_ylabel('E-YOLO', fontdict={'size': 8})
        else:
            axs[idx, 0].set_ylabel('YOLOv3', fontdict={'size': 8})
        axs[idx, 0].imshow(original_list[idx][..., ::-1])

        if idx == rows - 1:
            axs[idx, 1].set_xlabel('Level 1 (Small)', fontdict={'size': font_size})
        axs[idx, 1].imshow(array1_list[idx][..., ::-1])

        if idx == rows - 1:
            axs[idx, 2].set_xlabel('Level 2 (Medium)', fontdict={'size': font_size})
        axs[idx, 2].imshow(array2_list[idx][..., ::-1])

        if idx == rows - 1:
            axs[idx, 3].set_xlabel('Level 3 (Large)', fontdict={'size': font_size})
        axs[idx, 3].imshow(array3_list[idx][..., ::-1])

    # plt.savefig(os.path.join(base_path_eyolo, 'dst.jpg'))
    fig.tight_layout()
    plt.subplots_adjust(top=0.981, bottom=0.019, left=0.038, right=0.981, hspace=0.0, wspace=0.114)
    plt.show()


images = os.listdir(base_path_mmyolo)
images.sort()
ori_images = [image for image in images if 'anchor' not in image]

anchor_1 = [image for image in images if 'anchor1' in image]
anchor_2 = [image for image in images if 'anchor2' in image]
anchor_3 = [image for image in images if 'anchor3' in image]


for idx in range(len(ori_images)):
    anchor_1_img = cv2.imread(os.path.join(base_path_mmyolo, anchor_1[idx]), 0)
    anchor_1_img = cv2.resize(anchor_1_img, (anchor_1_img.shape[1] * 8, anchor_1_img.shape[0] * 8))
    anchor_1_img = np.mat(anchor_1_img, dtype=np.float)
    anchor_1_img = np.mat(anchor_1_img * color_len // 256, dtype=np.uint8)

    anchor_2_img = cv2.imread(os.path.join(base_path_mmyolo, anchor_2[idx]), 0)
    anchor_2_img = cv2.resize(anchor_2_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    anchor_2_img = np.mat(anchor_2_img, dtype=np.float)
    anchor_2_img = np.mat(anchor_2_img * color_len // 256, dtype=np.uint8)

    anchor_3_img = cv2.imread(os.path.join(base_path_mmyolo, anchor_3[idx]), 0)
    anchor_3_img = cv2.resize(anchor_3_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    anchor_3_img = np.mat(anchor_3_img, dtype=np.float)
    anchor_3_img = np.mat(anchor_3_img * color_len // 256, dtype=np.uint8)

    original_img = cv2.imread(os.path.join(base_path_mmyolo, ori_images[idx]))
    original_img_0 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    original_img_1 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    original_img_2 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    original_img_3 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    # print(img.shape, mask_img.shape, original_img_.shape)
    alpha = 0.3
    array1 = cv2.addWeighted(original_img_1, alpha, anchor2mask(anchor_1_img), 1 - alpha, 0)
    array2 = cv2.addWeighted(original_img_2, alpha, anchor2mask(anchor_2_img), 1 - alpha, 0)
    array3 = cv2.addWeighted(original_img_3, alpha, anchor2mask(anchor_3_img), 1 - alpha, 0)
    print('finished', idx + 1, '/', len(ori_images))
    # os.system('rm -f %s %s %s' % (os.path.join(base_path_eyolo, anchor_1[idx]),
    #                               os.path.join(base_path_eyolo, anchor_2[idx]),
    #                               os.path.join(base_path_eyolo, anchor_3[idx])))
    ori_list.append(original_img_0)
    arr1_list.append(array1)
    arr2_list.append(array2)
    arr3_list.append(array3)



images = os.listdir(base_path_eyolo)
images.sort()
ori_images = [image for image in images if 'anchor' not in image]



anchor_1 = [image for image in images if 'anchor1' in image]
anchor_2 = [image for image in images if 'anchor2' in image]
anchor_3 = [image for image in images if 'anchor3' in image]


for idx in range(len(ori_images)):
    anchor_1_img = cv2.imread(os.path.join(base_path_eyolo, anchor_1[idx]), 0)
    anchor_1_img = cv2.resize(anchor_1_img, (anchor_1_img.shape[1] * 8, anchor_1_img.shape[0] * 8))
    anchor_1_img = np.mat(anchor_1_img, dtype=np.float)
    anchor_1_img = np.mat(anchor_1_img * color_len // 256, dtype=np.uint8)

    anchor_2_img = cv2.imread(os.path.join(base_path_eyolo, anchor_2[idx]), 0)
    anchor_2_img = cv2.resize(anchor_2_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    anchor_2_img = np.mat(anchor_2_img, dtype=np.float)
    anchor_2_img = np.mat(anchor_2_img * color_len // 256, dtype=np.uint8)

    anchor_3_img = cv2.imread(os.path.join(base_path_eyolo, anchor_3[idx]), 0)
    anchor_3_img = cv2.resize(anchor_3_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    anchor_3_img = np.mat(anchor_3_img, dtype=np.float)
    anchor_3_img = np.mat(anchor_3_img * color_len // 256, dtype=np.uint8)

    original_img = cv2.imread(os.path.join(base_path_eyolo, ori_images[idx]))
    original_img_0 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    original_img_1 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    original_img_2 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    original_img_3 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    # print(img.shape, mask_img.shape, original_img_.shape)
    alpha = 0.3
    array1 = cv2.addWeighted(original_img_1, alpha, anchor2mask(anchor_1_img), 1 - alpha, 0)
    array2 = cv2.addWeighted(original_img_2, alpha, anchor2mask(anchor_2_img), 1 - alpha, 0)
    array3 = cv2.addWeighted(original_img_3, alpha, anchor2mask(anchor_3_img), 1 - alpha, 0)
    print('finished', idx + 1, '/', len(ori_images))
    # os.system('rm -f %s %s %s' % (os.path.join(base_path_eyolo, anchor_1[idx]),
    #                               os.path.join(base_path_eyolo, anchor_2[idx]),
    #                               os.path.join(base_path_eyolo, anchor_3[idx])))
    ori_list.append(original_img_0)
    arr1_list.append(array1)
    arr2_list.append(array2)
    arr3_list.append(array3)

images = os.listdir(base_path_yolo)
images.sort()
ori_images = [image for image in images if 'anchor' not in image]

anchor_1 = [image for image in images if 'anchor1' in image]
anchor_2 = [image for image in images if 'anchor2' in image]
anchor_3 = [image for image in images if 'anchor3' in image]


for idx in range(len(ori_images)):
    anchor_1_img = cv2.imread(os.path.join(base_path_yolo, anchor_1[idx]), 0)
    anchor_1_img = cv2.resize(anchor_1_img, (anchor_1_img.shape[1] * 8, anchor_1_img.shape[0] * 8))
    anchor_1_img = np.mat(anchor_1_img, dtype=np.float)
    anchor_1_img = np.mat(anchor_1_img * color_len // 256, dtype=np.uint8)

    anchor_2_img = cv2.imread(os.path.join(base_path_yolo, anchor_2[idx]), 0)
    anchor_2_img = cv2.resize(anchor_2_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    anchor_2_img = np.mat(anchor_2_img, dtype=np.float)
    anchor_2_img = np.mat(anchor_2_img * color_len // 256, dtype=np.uint8)

    anchor_3_img = cv2.imread(os.path.join(base_path_yolo, anchor_3[idx]), 0)
    anchor_3_img = cv2.resize(anchor_3_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    anchor_3_img = np.mat(anchor_3_img, dtype=np.float)
    anchor_3_img = np.mat(anchor_3_img * color_len // 256, dtype=np.uint8)

    original_img = cv2.imread(os.path.join(base_path_yolo, ori_images[idx]))
    original_img_0 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    original_img_1 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    original_img_2 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    original_img_3 = cv2.resize(original_img, (anchor_1_img.shape[1], anchor_1_img.shape[0]))
    # print(img.shape, mask_img.shape, original_img_.shape)
    alpha = 0.3
    array1 = cv2.addWeighted(original_img_1, alpha, anchor2mask(anchor_1_img), 1 - alpha, 0)
    array2 = cv2.addWeighted(original_img_2, alpha, anchor2mask(anchor_2_img), 1 - alpha, 0)
    array3 = cv2.addWeighted(original_img_3, alpha, anchor2mask(anchor_3_img), 1 - alpha, 0)
    print('finished', idx + 1, '/', len(ori_images))
    # os.system('rm -f %s %s %s' % (os.path.join(base_path_eyolo, anchor_1[idx]),
    #                               os.path.join(base_path_eyolo, anchor_2[idx]),
    #                               os.path.join(base_path_eyolo, anchor_3[idx])))
    ori_list.append(original_img_0)
    arr1_list.append(array1)
    arr2_list.append(array2)
    arr3_list.append(array3)





len_ori = len(ori_list)
indi = len_ori // 3
a = np.array([[i, indi+i, 2*indi+i] for i in range(indi)])
a = a.reshape(-1)

ori_list = np.array(ori_list)[a]
arr1_list = np.array(arr1_list)[a]
arr2_list = np.array(arr2_list)[a]
arr3_list = np.array(arr3_list)[a]

print(ori_list.shape)
show_plts(ori_list, arr1_list, arr2_list, arr3_list)

