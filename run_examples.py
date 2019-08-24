"""
Runs the program for the images in ./data/ which includes the example images from the original author and a few extras
ones.
"""

import os

art_styles = [1.203930, 5.323582, 4.234591, 5.498451, 4.910163, 9.982656, 5.731922, 9.862515, 5.019980, 8.189465,
              1.340035, 9.331765, 7.318527, 5.106254, 1.295220, 1.001454, 1.018541, 1.000000, 4.999322, 1.002947,
              1.000424, 4.953209, 4.997079, 9.995778, 4.800576, 5.062284, 1.001287, 5.000851, 4.977097, 2.303986,
              5.017763, 4.983986, 4.999594, 1.512768, 1.036211]

if __name__ == "__main__":
    for i in range(35):
        print("------------------------------------------------\n")
        print("Image {}:\n".format(i))
        os.system('deep_painterly_harmonization.py '
                  '-c data/{}_naive.jpg '
                  '-s data/{}_target.jpg '
                  '-o output/output_{}.png '
                  '-m data/{}_c_mask.jpg '
                  '--num_cores 12 '
                  '--art_stylization {}'.format(i, i, i, i, art_styles[i], i))
