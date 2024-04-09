from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
def plot(model,img):
    plot = model.predict_tile(img, return_plot = True,iou_threshold=0.4, patch_size=250)
    plt.imshow(plot[:,:,::-1])

def count(model,img):
    cnt = model.predict_tile(img, return_plot = True,iou_threshold=0.4, patch_size=250)
    return len(cnt)

def main():
    from deepforest import main
    from deepforest import get_data
    model = main.deepforest()
    model.use_release()
    sample_image_path = get_data("drone-view-forest.jpg")
    plot(model,sample_image_path)
    print("Number of Trees in the Image :",count(model,sample_image_path))

if __name__ == "__main__":
    main()