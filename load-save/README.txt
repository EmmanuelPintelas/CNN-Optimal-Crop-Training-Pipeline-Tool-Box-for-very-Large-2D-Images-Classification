In this project we read/load the huge raw images (tiff format) from a folder 
and then resize them and save them in jpg format.
In order to load them we utilize also open-slide tools. 
We also provide some functions for detecting the most informative
tiles (crops) and save them also as crops. 
A croped image can focus on local areas and thus the resolution of important regions
is much higher. Feeding them in such a way can drastically boost the performance of a CNN model.