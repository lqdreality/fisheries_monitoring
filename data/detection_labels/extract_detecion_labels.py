import csv
import os, glob
import json


def open_files(folder_path):
    boxes = []
    for filename in glob.glob(os.path.join(folder_path, '*.json')):
        with open(filename, 'r') as f:
            data = json.load(f)
            # Go through every image in the file
            for img in data:
                name = img[u'filename']  # Name of the file
                fishes = img[u'annotations']  # List with dicts of every box

                # Extract the location of each box (x, y, width, height) and add the name of the file to it
                boxes += [[name, box[u'x'], box[u'y'], box[u'width'], box[u'height']] for box in fishes]

    return boxes


def write_csv(data, fname):
    with open(fname, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in data:
            writer.writerow(line)


def main():
    boxes = open_files("data")

    write_csv(boxes, "boxes.csv")

if __name__ == '__main__':
    main()
