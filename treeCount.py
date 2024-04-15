import matplotlib.pyplot as plt
import warnings
from  PIL import Image
warnings.filterwarnings("ignore")
def plot(model1,img):
    plot = model1.predict_tile(img, return_plot = True,iou_threshold=0.4, patch_size=250)
    print(type(plot))
    plt.imshow(plot[:,:,::-1])

def count(model1,img):
    cnt = model1.predict_tile(img, return_plot = True,iou_threshold=0.4, patch_size=250)
    im = Image.fromarray(cnt)
    im.save("image.jpg")
    return len(cnt)

def main():
    from deepforest import main
    from deepforest import get_data
    model1 = main.deepforest()
    model1.use_release()
    sample_image_path = get_data("treeImage.jpg")
    plot(model1,sample_image_path)
    print("Number of Trees in the Image :",count(model1,sample_image_path))

if __name__ == "__main__":
    main()
