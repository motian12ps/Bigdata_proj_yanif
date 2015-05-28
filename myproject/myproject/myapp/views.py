# -*- coding: utf-8 -*-
from django.shortcuts import render_to_response
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse

from myproject.myapp.models import Document
from myproject.myapp.forms import DocumentForm


import sys, getopt
import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.mlab import PCA as mlabPCA
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from PIL import Image
import pickle

def show(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile = request.FILES['docfile'])
            newdoc.save()
            print newdoc.docfile

            i = Image.open(newdoc.docfile)
            i = i.convert('RGB')
            pix=i.load()
            w=i.size[0]
            h=i.size[1]
            RT = numpy.zeros(w*h)
            BT = numpy.zeros(w*h)
            GT = numpy.zeros(w*h)
            for i in range(h):
                for j in range(w):
                    r,g,b = pix[j,i]
                    RT[i*32 + j] = r
                    GT[i*32 + j] = g
                    BT[i*32 + j] = b

            grayMatrixT = (RT*0.2989+GT*0.5870+BT*0.1140)
            #Normalize
            grayMatrixT -= grayMatrixT.mean()

            dict = unpickle('data_batch_1')
            dt = unpickle('batches.meta')
            print dt
            size = len(dict['data'])
            featureSize = len(dict['data'][0])/3
            grayMatrix = numpy.zeros((size, featureSize))
            normalRGB = numpy.zeros((size, featureSize,3))

            #Compute intensity and normalize
            items = list(dict.items())
            dataItem = items[0]
            labels = numpy.array(dict['labels'])
            print labels
            dataMatrix = numpy.array(dataItem[1])
            R = dataMatrix[:,:1024]
            G = dataMatrix[:,1024:2048]
            B = dataMatrix[:,2048:]
            #Construct original picture
            normalRGB = numpy.rollaxis(numpy.asarray([R,G,B]), 0,3)
            #Construct intensity Array
            grayMatrix = (R*0.2989+G*0.5870+B*0.1140)
            #Normalize
            grayMatrix -= grayMatrix.mean(axis=1)[:, None]

            #Apply PCA
            train(grayMatrix, featureSize, labels, grayMatrixT)

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('myproject.myapp.views.show'))
    else:
        form = DocumentForm() # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render_to_response(
        'myapp/show.html',
        {'documents': documents, 'form': form},
        context_instance=RequestContext(request)
    )


def plot(data): 
    plt.imshow(data, cmap = cm.Greys_r)
    plt.show()

def getArguments(argv):
    if len(sys.argv) != 2:
        print 'Please pass training file'
        sys.exit()
    return str(sys.argv[1])

def train(matrix, featureSize, labels, predictor):

    cov_mat = numpy.cov(matrix.T)
    print cov_mat.shape
    eig_val_cov, eig_vec_cov = numpy.linalg.eig(cov_mat)# Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(numpy.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    matrix_w = eig_pairs[0][1].reshape(featureSize,1)
    for i in range(200):
        matrix_w = numpy.hstack((matrix_w, eig_pairs[i+1][1].reshape(featureSize,1)))
    print matrix_w.shape

    transformed = matrix.dot(matrix_w)
    print transformed.shape
    #Compute cov matrix
    if os.path.isfile('svm.model'):
        print 'Loading Model file...'
        #Load models from file
        with open('svm.model', 'rb') as file:
            Z = pickle.load(file)
    else:
        #Start to train SVM
        Z = OneVsRestClassifier(SVC(kernel="rbf")).fit(transformed, labels)
        with open('svm.model', 'wb') as file:
            pickle.dump(Z, file)


    recData = transformed.dot(matrix_w.T) + matrix.mean(axis=1)[:, None]
    j = Image.fromarray(recData[0].reshape((32,32)))
    newdoc = Document(docfile = "documents/pca.png")
    j = j.convert('L')
    j.save("myproject/media/documents/pca.png");
    newdoc.save();

    Z =  Z.predict(predictor.dot(matrix_w))
    res = Z[0]
    print res
    correct = 0.0
    for x in range(len(Z)):
        if labels[x] == Z[x]:
            correct = correct +1

    print correct/len(Z)
    #recData = transformed.dot(matrix_w.T) + matrix.mean(axis=1)[:, None]
    #plot(recData[0].reshape((32,32)))
            

def unpickle(file):
    import cPickle
    module_dir = os.path.dirname(__file__)  # get current directory
    file_path = os.path.join(module_dir, file)
    fo = open(file_path, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict