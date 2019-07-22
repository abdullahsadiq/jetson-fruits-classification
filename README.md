# Classifying fruits on Jetson Nano
Classification of fruits with transfer learning using Tensorflow on Jetson Nano.

Tested on Jetson Nano but should work on other platforms as well.

I would suggest that if you're a beginnner you should go through the Hello AI World [example](https://github.com/dusty-nv/jetson-inference) before continuing; it will give you a great introduction as well as automating the install of dependencies so you don't have to worry about those later. You will also need this if you will be making your own dataset.

Before moving on, you should make and mount a swap file (at least 4GB). It will make your life a lot easier and is a must for this to work. More details [here](https://support.rackspace.com/how-to/create-a-linux-swap-file/).

```
sudo fallocate -l 4G /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
```

For making the change permanent you will need to add this line at the end of `/etc/fstab` :

```
/mnt/4GB.swap  none  swap  sw 0  0
```

You can run `sudo tegrastats` to confirm if the swap file is mounted.

Then clone my project repository with:

```
git clone https://github.com/abdullahsadiq/jetson-fruits-classification.git
```

For classifying anything we need a proper dataset. Having tested out the ones available online (FIDS30 and Fruits360), I wasn't satisfied with the result so I made my own dataset, a small one with 6 classes and a total of 600 images (100 for each class). I used the `camera-capture` utility in the Hello AI World example to capture images.

You can find my dataset in the *fruits-dataset* folder in *jetson-fruits-classification*; you can either use that, another dataset from the Internet or simply make your own.

These are the classes in my dataset:

```
	• Apple/
	• Beetroot/
	• Dates/
	• Mango/
	• Orange/
	• Pomegranate/
```

## Making your own Dataset

You can find the details of using the `camera-capture` utility [here](https://github.com/dusty-nv/jetson-infehttps://github.crence/blob/master/docs/pytorch-collect.md).

Make a folder which will store your new dataset and create a file in the root of this folder called *labels.txt*, where you should add the names of your dataset's classes (one on each line).

```
mkdir my-fruits-dataset
cd my-fruits-dataset
gedit labels.txt
```

Remember to save the file after adding the names.

Head over to `jetson-inference/build/tools/camera-capture` and run:

```
make
```

This will make an executable file called `camera-capture` which will be located in `jetson-inference/build/aarch64/bin`. If you're still in the same directory as above run:

```
cd ../../aarch64/bin/
./camera-capture
```

This will open the camera-capture utility. I will assume that you are using the CSI camera on the Jetson Nano; for other cameras read the link I mentioned earlier which has the details of the `camera-capture` utility. You need to select your dataset's directory and your *labels.txt* file created earlier. Once the labels are loaded it will show you the classes and you can continue to take images. You should take images for each class in the 'train' folder only; there is no need to have separate 'train' and 'val' folders because we will be retraining using Tensorflow and it gives an error if we organize our data in this way. Use the spacebar shortcut to capture images as it will make the job much quicker. Capture at least 100 images for each class to get good results when retraining.

Once you are done, move all your classes out of the 'train' folder (assuming the images you took are present there) and place them at the root of the `my-fruits-dataset` so that the directory structure looks like this:

```
‣ my-fruits-dataset/
	• class-A/
	• class-B/
	• class-C/
	• labels.txt
```

Note that *labels.txt* won't exactly be required from this point onwards.

Now you have created a dataset with your own images structured for retraining in Tensorflow and you can move on to retraining a model.

## Re-training using Tensorflow

Note that these files are not written by me, they are the example files for the official Tensorflow tutorial for retraining which can be found [here](https://www.tensorflow.org/hub/tutorials/image_retraining).

If you want to just test out my retrained model which is included in the project repository, you can skip this section and move on to the next section for inference.

When retraining a dataset, you have the choice to do it on the Jetson Nano or on a host PC. The steps will be the same on both; you need to clone the repository and use the script for retraining, and once done the model will be saved as `/tmp/output_graph.pb` and `/tmp/output_labels.txt`.

Navigate to the `jetson-fruits-classification` root and open a terminal from there. Run the following while replacing the directory of the dataset you wish to train:

```
python3 retrain/retrain.py --image_dir fruits-dataset/
```

This will start the training process on my dataset which took about 1 hour on the Jetson Nano but you could do it on a host PC and transfer the output model back to the Jetson Nano for inference. Note that if you use a host PC for retraining the model and Jetson Nano for inference, you need to make sure that the Tensorflow version installed is the same on both systems otherwise it won't work.

If you get errors about any modules not found simply install them with `pip3` and re-run the script.

Once the retraining is complete move and save the two output files from the *tmp* folder because they will be deleted when your system is turned off, and move on to the next step for inference.

## Inference

You will need some images of fruits to test the model. I have included some from Google Images in the *test-images* folder of the repository. You can either use the model which you trained earlier or the one which I have already trained and is present in the repository in the *model* folder. Assuming you are still in the root of the `jetson-fruits-classification`, just copy the model you want to test to the *retrain* folder and run:

```
python3 retrain/label_image.py --graph=retrain/my_output_graph.pb --labels=retrain/my_output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=test-images/apple.jpeg
```

If you want to use another model located anywhere else just make sure to give it's proper location.

After a while you should see output like this:

```
apple 0.9856818
orange 0.005912187
dates 0.002886865
pomegranate 0.0026501939
mango 0.0014653547
```

This means that the model accurately recognized the image as an apple. At this moment, you should test it with other images, either from the *test-images* folder or from the Internet by replacing the directory for the test image.

## Further improvements

To further improve the model we need to optimize it with TensorRT which will reduce the time taken in inferencing.

By default the retraining script uses Inception_v3 but we can use more lighter model architectures so that the inference time is reduced on the Jetson Nano.
