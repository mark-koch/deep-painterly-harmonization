# Deep Painterly Harmonization

A TensorFlow implementation of "[Deep Painterly Harmonization](https://arxiv.org/abs/1804.03189)".

<p align='center'>
  <img src='data/4_target.jpg' width='290'/>
  <img src='data/4_naive.jpg' width='290'/>
  <img src='output/output_4.png' width='290'/>
</p>


## Setup

Requirements:
* Tensorflow (tested with version 1.13)
* OpenCV (if you want to run the post processing you will also need to install the contrib modules)
* scikit-learn
* SciPy
* NumPy

You also need to download the VGG19 model weights from [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat).


## Usage

Run ``deep_painterly_harmonization.py -c content.png -s painting.png -o output.png``.

Optional Arguments:
* ``-m``: Mask image (a black image with the inserted objects marked in white). If not provided, the program generates one 
automatically.
* ``-a``: Hyperparameter regarding how stylized the target painting is. Corresponds to the output of the painting estimator 
in the paper.
* ``--vgg``: Path of the the VGG19 weights.
* ``--max_size``: Maximum height or width of the images. Default is set to 700.
* ``--num_cores``: Number of processes to use for the post processing.
* ``--only``: Can be used to run only parts of the image generation pipeline (``pass1``, ``pass2``, ``post``). For example to 
skip the second pass, you could run the program with ``--only pass1 post`` . When skipping the first pass or only running the
post processing, you might want to provide the output of the previous skipped stage with ``--inter_res``.

To see a description of the other arguments, run ``deep_painterly_harmonization.py -h``.



## Notes

* There were a few cases where the original code of the authors contradicted the statements in their paper. In these cases I
tried to follow the code.
* The histogram loss for the second pass is implemented but does not seem to function properly and is deactivated for now.
* The painting estimater to classify art styles is not implemented yet. You can set the style value with the parameter ``-a``.


## Acknowledgement

* See the [original code](https://github.com/luanfujun/deep-painterly-harmonization) for the paper.
* This [blog post](https://sgugger.github.io/deep-painterly-harmonization.html) by Sylvain Gugger was also really helpful.


## Examples

These are the results for a few of the original examples by the authors. You can find all examples in the ``output/`` directory.
To generate these examples, use the script ``run_examples.py``.

<p align='center'>
  <img src='data/13_target.jpg' width='290'/>
  <img src='data/13_naive.jpg' width='290'/>
  <img src='output/output_13.png' width='290'/>
</p>

<p align='center'>
  <img src='data/15_target.jpg' width='290'/>
  <img src='data/15_naive.jpg' width='290'/>
  <img src='output/output_15.png' width='290'/>
</p>

<p align='center'>
  <img src='data/18_target.jpg' width='290'/>
  <img src='data/18_naive.jpg' width='290'/>
  <img src='output/output_18.png' width='290'/>
</p>

<p align='center'>
  <img src='data/19_target.jpg' width='290'/>
  <img src='data/19_naive.jpg' width='290'/>
  <img src='output/output_19.png' width='290'/>
</p>

<p align='center'>
  <img src='data/0_target.jpg' width='290'/>
  <img src='data/0_naive.jpg' width='290'/>
  <img src='output/output_0.png' width='290'/>
</p>

<p align='center'>
  <img src='data/16_target.jpg' width='290'/>
  <img src='data/16_naive.jpg' width='290'/>
  <img src='output/output_16.png' width='290'/>
</p>

<p align='center'>
  <img src='data/5_target.jpg' width='290'/>
  <img src='data/5_naive.jpg' width='290'/>
  <img src='output/output_5.png' width='290'/>
</p>

<p align='center'>
  <img src='data/6_target.jpg' width='290'/>
  <img src='data/6_naive.jpg' width='290'/>
  <img src='output/output_6.png' width='290'/>
</p>

<p align='center'>
  <img src='data/10_target.jpg' width='290'/>
  <img src='data/10_naive.jpg' width='290'/>
  <img src='output/output_10.png' width='290'/>
</p>

<p align='center'>
  <img src='data/21_target.jpg' width='290'/>
  <img src='data/21_naive.jpg' width='290'/>
  <img src='output/output_21.png' width='290'/>
</p>

<p align='center'>
  <img src='data/24_target.jpg' width='290'/>
  <img src='data/24_naive.jpg' width='290'/>
  <img src='output/output_24.png' width='290'/>
</p>

<p align='center'>
  <img src='data/31_target.jpg' width='290'/>
  <img src='data/31_naive.jpg' width='290'/>
  <img src='output/output_31.png' width='290'/>
</p>

<p align='center'>
  <img src='data/34_target.jpg' width='290'/>
  <img src='data/34_naive.jpg' width='290'/>
  <img src='output/output_34.png' width='290'/>
</p>
