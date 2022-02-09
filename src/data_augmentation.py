import cv2
import imageio
import imgaug as ia
import laerte
from laerte import anticlee
from laerte import autolycos
from laerte import eurytos
from laerte import telemaque
from laerte import calypso
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

layer_list = anticlee()
dict_label_clr = laerte.sisyphe()
picture_list = autolycos()
val_picture_list = eurytos()
mask_list = telemaque()
val_mask_list = calypso()


def wart_hog():
    """C’est la fonction de data augmentation qui va chercher les images et les masques, les transforme puis
    les sauvegardes dans le même répertoire sous le format jpg."""
    image_str_list = [picture_list, val_picture_list]
    mask_str_list = [mask_list, val_mask_list]
    for num_pack in range(2):
        it = 0
        for image_str, mask_str in zip(image_str_list[num_pack], mask_str_list[num_pack]):
            """Contrôle de la correspondance entre l’image et le masque par assert"""
            assert (image_str.split('/')[-1].split('_leftImg8bit')[0] == mask_str.split('/')[-1]
                    .split('_gtFine_polygons_octogroups')[0]) | \
                   (image_str.split('/')[-1].split('_leftImg8bit')[0] == mask_str
                    .split('/')[-1].split('_gtFine_polygons_octogroups')[0])
            if it % 75 == 0:
                print(f'traiter : {it}\nrestant : {it - len(image_str_list)}')
            image = imageio.imread(image_str)
            segmap = imageio.imread(mask_str)
            seq = iaa.Sequential([
                iaa.Affine(rotate=(-25, 25)),
                iaa.AdditiveGaussianNoise(scale=(15, 60)),
                iaa.Crop(percent=(0, 0.2)),
                iaa.Dropout([0.05, 0.2]),
                iaa.Sharpen((0.0, 1.0)),
                iaa.ElasticTransformation(alpha=50, sigma=5)
            ], random_order=True)
            segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
            image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
            img_name = image_str[-4] + '.jpg'  # "image_" + str(num_pack) + '_' + str(it) + ".jpg"
            image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_name, image_aug)
            segmap_name = mask_str[-4] + '.jpg'  # "segmap_" + str(num_pack) + '_' + str(it) + ".jpg"
            segmap_aug = segmap_aug.draw(size=image_aug.shape[:2])[0]
            imageio.imwrite(segmap_name, segmap_aug)
            it += 1


def pig():
    """Cette fonction permet de produire une image de ce que la data augmentation peut
    produire."""
    image_str_list = [picture_list, val_picture_list]
    mask_str_list = [mask_list, val_mask_list]
    for num_pack in range(2):
        it = 0
        images = []
        masks = []
        images_aug = []
        segmaps_aug = []
        for image_str, mask_str in zip(image_str_list[num_pack][:5], mask_str_list[num_pack][:5]):
            assert (image_str.split('/')[-1].split('_leftImg8bit')[0] == mask_str.split('/')[-1]
                    .split('_gtFine_polygons_octogroups')[0]) | \
                   (image_str.split('/')[-1].split('_leftImg8bit')[0] == mask_str
                    .split('/')[-1].split('_gtFine_polygons_octogroups')[0])
            if it % 75 == 0:
                print(f'traiter : {it}\nrestant : {it - len(image_str_list)}')
            image = imageio.imread(image_str)
            images.append(image)
            segmap = imageio.imread(mask_str)
            seq = iaa.Sequential([
                iaa.Affine(rotate=(-25, 25)),
                iaa.AdditiveGaussianNoise(scale=(15, 60)),
                iaa.Crop(percent=(0, 0.2)),
                iaa.Dropout([0.05, 0.2]),
                iaa.Sharpen((0.0, 1.0)),
                iaa.ElasticTransformation(alpha=50, sigma=5)
            ], random_order=True)
            segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
            masks.append(segmap)

            images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
            images_aug.append(images_aug_i)
            segmaps_aug.append(segmaps_aug_i)
            img_name = "image_" + str(num_pack) + '_' + str(it) + ".jpg"  # image_str[-4] + '.jpg'
            images_aug_i = cv2.cvtColor(images_aug_i, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_name, images_aug_i)
            it += 1
        cells = []
        for image, segmap, image_aug, segmap_aug in zip(images, masks, images_aug, segmaps_aug):
            cells.append(image)
            cells.append(segmap.draw_on_image(image)[0])
            cells.append(image_aug)
            cells.append(segmap_aug.draw_on_image(image_aug)[0])
            cells.append(segmap_aug.draw(size=image_aug.shape[:2])[0])

        grid_image = ia.draw_grid(cells, cols=5)
        imageio.imwrite("example_segmaps.jpg", grid_image)
    print("Augmented")


def main():
    #pig()  # Pour faire un graphique de présentation de la data_augmentation
    wart_hog() # Pour faire la data augmentation sur les répertoire train et val. Test est laissé de côté car
    # les masques sont tous opaques.


if __name__ == "__main__":
    main()
