#Import tkinter library
from tkinter import *
#Create an instance of Tkinter frame or window
win= Tk()
#Set the geometry of tkinter frame
win.geometry("750x250")
#Make the window sticky for every case
win.grid_rowconfigure(0, weight=1)
win.grid_columnconfigure(0, weight=1)
#Create a Label
label=Label(win, text="This is a Centered Text",font=('Aerial 15 bold'))
label.grid(row=1, column=0)
label.grid_rowconfigure(1, weight=1)
label.grid_columnconfigure(1, weight=1)
win.mainloop()

#Random forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

#SVM
from sklearn.svm import SVC
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

#Neural Networks
from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
nn_model.fit(X_train, y_train)