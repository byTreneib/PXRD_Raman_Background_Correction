from tkinter import filedialog
from os import getcwd

files = filedialog.askopenfilenames(initialdir=getcwd(), title="Select Files")

for file in files:
    with open(file) as readfile:
        content = readfile.read()

    with open(file + '.txt', 'w+') as writefile:
        writefile.write(content)
