### TF2 resnet152 implementation

## To train the ResNet on your own dataset, you can put the dataset under the folder original dataset, and the directory should look like this:
|——original dataset
   |——class_name_0
   |——class_name_1
   |——class_name_2
   |——class_name_3
Run the script split_dataset.py to split the raw dataset into train set, valid set and test set.
Change the corresponding parameters in config.py.
