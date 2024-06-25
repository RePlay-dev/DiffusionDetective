# Getting the training Data
To obtain the training data that is being used by the `training.ipynb` notebook you need to take the following steps:

## Download the Synthbuster dataset
Download the Synthbuster dataset from https://zenodo.org/records/10066460 and extract the subfolders containg the images to the `data/full/fake/synthbuster` folder.

## Download the Flickr30K dataset
Download the Flickr30K dataset from https://shannon.cs.illinois.edu/DenotationGraph/ . The link to request the dataset is at the bottom of the page. Then extract the images to the `data/full/real/flickr` folder.

## Download the RAISE-1k datset
Download the RAISE-1k dataset from http://loki.disi.unitn.it/RAISE/download.html . After downloading all the images convert them to resized png's by running the following bash script inside the folder containing the images. ImageMagick must be installed on the system to use the `convert` command.

```bash
mkdir png
find . -name "*.TIF" | parallel --bar convert {} png/{/.}.png
find png -name "*-1.png" -exec rm {} \;
find png -name "*.png" | parallel --bar convert {} -resize "1024x1024^" -quality 100 resized/{/}
```

Afterwards move the images in the `resized` folder to the `data/full/real/raise` folder.

## Download the ImageNet-1k-valid dataset
Download the ImageNet-1k-valid dataset from https://www.kaggle.com/datasets/sautkin/imagenet1kvalid . Then extract the subfolders containing the images to the `data/full/real/imagenet1k` folder.

## Download the Stable ImageNet-1K dataset
Download the Stable ImageNet-1K dataset from https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k . Then extract the subfolders containing the images to the `data/full/fake/stable-imagenet1k` folder.

## Split dataset
Now the folder structure should be like this:

```
data
├── full
│   ├── fake
│   │   ├── stable-imagenet1k
│   │   └── synthbuster
│   │       ├── dalle2
│   │       ├── dalle3
│   │       ├── firefly
│   │       ├── glide
│   │       ├── midjourney-v5
│   │       ├── stable-diffusion-1-3
│   │       ├── stable-diffusion-1-4
│   │       ├── stable-diffusion-2
│   │       └── stable-diffusion-xl
│   └── real
│       ├── flickr
│       ├── imagenet1k
│       └── raise
```

Now run the `split_dataset.py` script to split the dataset into a training, test, and validation set.

```bash
python split_dataset.py full/
```

Or use the following command to move the images instead of copying them in order to save disk space.

```bash
python split_dataset.py full/ --move
```


## Download the Midjourney CIFAKE-Inspired Dataset
Since the Midjourney CIFAKE-Inspired Dataset is already split into a test, training and validation set they need to be placed into the folder structure manually. Download the dataset from https://www.kaggle.com/datasets/mariammarioma/midjourney-cifake-inspired and then move the contents in the following way.

- Midjourney/test/FAKE/ to test/fake/midjourney_cifake/
- Midjourney/train/FAKE/ to training/train/fake/midjourney_cifake/
- Midjourney/valid/FAKE/ to training/valid/fake/midjourney_cifake/

Alternatively you can run the bash script `move_midjourney_cifake.sh` to move the files. Delete the `Midjourney` afterwards.
