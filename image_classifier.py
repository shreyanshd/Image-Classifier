import matplotlib.pylab as pl
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import base64
from StringIO import StringIO
from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os

from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

STANDARD_SIZE = (300, 167)

def download_images(label):

	image_type = label
	query = label
	url = "http://www.bing.com/images/search?q=" + query + "&qft=+filterui:color2-bw+filterui:imagesize-large&FORM=R5IR3"

	soup = BeautifulSoup(requests.get(url).text)

	images = [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})]

	for img in images:
	    raw_img = urllib2.urlopen(img).read()
	    cntr = len([i for i in os.listdir("images") if image_type in i]) + 1
	    f = open("images/" + image_type + "_"+ str(cntr) + ".jpg", 'wb')
	    f.write(raw_img)
	    f.close()

	print("Downloaded {} images of label {} and saved to images folder.".format(len(images),label))



def img_to_matrix(filename, verbose=False):

    img = PIL.Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
 
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def convert_image_to_array():
	print "converting images to array...."
	image_dir = "images/"
	images = [image_dir+ f for f in os.listdir(image_dir)]
	labels = ["sphere" if "sphere" in f.split('/')[-1] else "cube" for f in images]

	data = []

	for image in images:
		img = img_to_matrix(image)
		img = flatten_image(img)
		data.append(img)

	data = np.array(data)
	print "Done."
	return data,labels

def create_training_and_test_set(data , labels):
	print "Creating training set..."
	is_train = np.random.uniform(0,1,len(data)) <= 0.7
	y = np.where(np.array(labels)=='sphere' , 1 ,0)
	train_x , train_y = data[is_train] , y[is_train]
	print "Done."
	print "Creating test set..."
	test_x , test_y = data[is_train == False] , y[is_train == False]
	print "Done."
	return train_x , train_y , test_x , test_y ,y


def plot_for_2d(data , y):
	print "Reducing dimension to 2D for visualization...."
	pca = RandomizedPCA(n_components=2)
	X = pca.fit_transform(data)
	df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label":np.where(y==1, "Sphere", "cube")})
	colors = ["red", "yellow"]
	print "Displaying plot...."
	for label, color in zip(df['label'].unique(), colors):
		mask = df['label'] == label
		pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
	pl.show()
	print "Done."


def dimentionality_reduction(train_x , test_x):
	print "Dimentionality reduction to 10D on training and test data...."
	pca = RandomizedPCA(n_components=10)
	train_x = pca.fit_transform(train_x)
	test_x = pca.transform(test_x)
	print "Done."
	return train_x , test_x


def knn_classifier(train_x , train_y):
	print "Training knn_classifier...."
	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(train_x , train_y)
	print "Done."
	return knn

def evaluate(knn , test_x , test_y):
	print "Evaluating the classifier...."
	ans = pd.crosstab(test_y, knn.predict(test_x), rownames=["Actual"], colnames=["Predicted"])
	return ans


def summary(data , labels , train_x , train_y , test_y , ans):
	print "\n"
	print "=============================== SUMMARY =============================="
	print "\n"
	print "Total data : {} images:".format(len(data))
	print "28 {}".format(labels[0])
	print "28 {}".format(labels[55])
	print "\n"
	print "Training data size: {} => {}%".format(len(train_y) , (len(train_y)*100/56))
	print "Test data size: {} => {}%".format(len(test_y) , (len(test_y)*100/56))
	print "\n"
	print "Algorithm for Dimentionality Reduction: RandomizedPCA"
	print "Dimentionality reduction: {}D to {}D".format(len(data[0]) , 10)
	print "\n"
	print "classifier: K-Nearest-Neighbour"
	print "Value of k in kNN: 15"
	print "\n"
	print "Classifier Evaluation Matrix:"
	print ans
	print "\n"
	acc = (ans[0][0] + ans[1][1])*100/(ans[0][0] + ans[0][1] + ans[1][0] + ans[1][1])
	print "Classifier accuracy = {}%".format(acc)
	print "\n"
	print "======================================================================="


#def knn(train_x , train_y):


#print "Downloading data...."
#download_images("sphere")
#download_images("cube")

data , labels = convert_image_to_array()

train_x ,train_y , test_x, test_y, y = create_training_and_test_set(data , labels)

plot_for_2d(data ,y)

train_x , test_x = dimentionality_reduction(train_x , test_x)

knn = knn_classifier(train_x , train_y)

ans = evaluate(knn , test_x , test_y)

summary(data , labels , train_x , train_y , test_y , ans)

