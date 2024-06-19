import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# Example camera matrix (this should be your camera's intrinsic matrix)
cameraMatrix = np.array([[1000, 0, 320],
                         [0, 1000, 240],
                         [0, 0, 1]], dtype=np.float32)
img_path = "G:\\CNNs\\ce4\\descentimgs\\Choose\\"
base_img = "G:\\CNNs\\ce4\\descentimgs\\ce4split_03_4.jpg"

def project_point(point, M):
    point = np.array([[point[0]], [point[1]], [1]])
    new_point = np.dot(M, point)
    new_point /= new_point[2]
    return (new_point[0, 0], new_point[1, 0])


def cal_pose(kps1, kps2, img, point=None):
    """
    按照kps1与kps2的对应关系，将第二张图片映射到第一张图的局部
    """
    new_point = None
    M, status = cv2.findHomography(kps1, kps2, cv2.RANSAC, 5.0)
    if point:
        new_point = project_point(point, M)
    return new_point, cv2.warpPerspective(img, M, img.shape[:2][::-1])

def draw(x, y, *args):
    m = x * y
    for i in range(m):
        plt.subplot(x, y, i + 1), plt.title(args[i][0]), plt.imshow(args[i][1], cmap=("gray"))
    plt.show()

@np.vectorize
def fugai(x, y):
    return x if y == 0 else y

def circle(*args):
    """
    将传入点进行int化后按照原变量绘制圆形
    """
    x, y = args[1]
    x, y = int(x), int(y)
    args = list(args)
    args[1] = (x, y)
    args = tuple(args)
    return cv2.circle(*args)

def getPoint(ImgName:str, ):
    """
    输入图片名称，以及得到的匹配点坐标，输出其在大底图上的
    """
    row = ImgName.split("ce4split2048_")[1].split("_")[0]
    col = ImgName.split("ce4split2048_")[1].split("_")[1].split(".")[0]

    row, col = int(row), int(col)

    row_label = [1536, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 1536]
    col_label = [1536, 2048, 2048, 2048, 2048, 2048, 2048, 1536]

    return (sum(row_label[:row]), sum(col_label[:col]))


def Pixel2Coordinate(ImageCoordinate):
    经度 = [177.29, 177.80]
    纬度 = [45.31, 46.22]
    总尺寸 = [19627, 7800]

    row_ratio = ImageCoordinate[0] / 总尺寸[0]
    col_ratio = ImageCoordinate[1] / 总尺寸[1]

    latitude = (纬度[1] - 纬度[0]) * row_ratio + 纬度[0]
    longitude = (经度[1] - 经度[0]) * col_ratio + 经度[0]
    return (longitude, latitude)

def test1():
    kps1 = np.loadtxt("point0.txt")
    kps2 = np.loadtxt("point1.txt")
    img1 = cv2.imread(base_img, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (1024, 1024))
    img2 = cv2.imread(img_path + "ce4_all_01778.jpg", cv2.IMREAD_GRAYSCALE)[:1024, :1024]
    point = [512, 512]
    img2 = cv2.circle(img2, (512, 512), 10, 255, 3)
    newpos1, P = cal_pose(kps2, kps1, img2, point)
    newpos2, P2 = cal_pose(kps1, kps2, img1, point)
    print(newpos1, newpos2)

    P = circle(P, newpos1, 2, 255, -1)
    draw(2, 2, ("base_img", img1), ("waitimg", img2), ("P1", P), ("P2", P2))

if __name__ == '__main__':
    ImgDirectory = r"G:\CNNs\ce4\baseimgs\NAC_DTM_CHANGE4_M1303619844_140CM_split2048\\"
    InitPoint = getLUPoint(ImgDirectory, "ce4split2048_020_008.jpg")
    print(InitPoint)
    print(InitPoint[0] / 19627, InitPoint[1] / 7800)

    imgnames = os.listdir(ImgDirectory)
    img = cv2.imread(ImgDirectory + imgnames[0])

    base = cv2.imread(r"G:\CNNs\ce4\baseimgs\NAC_DTM_CHANGE4_M1303619844_140CM.tiff")
    draw(1, 2, ("00", img), ("base", base))


    
    
