import socket
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
image_count = 0
max_img_count = 3

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # AF_INET = IP, SOCK_STREAM = TCP

server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server.bind(('10.0.0.46', 1002))
server.listen()
print("Socket server opened. Waiting for connection..")

client_socket, client_address = server.accept()

print("Connected to client at address {}".format(client_address))

client_socket.settimeout(8)

while image_count < max_img_count:
    if image_count != (max_img_count - 1):
        file = open('server_{}.ppm'.format(image_count), "wb")
    else:
        file = open('server_{}.jpg'.format(image_count), "wb")
    try:
        image_chunk = client_socket.recv(2048)  # stream-based protocol
        while image_chunk:
            file.write(image_chunk)
            try:
                image_chunk = client_socket.recv(2048)
            except:
                print("recv timeout")
                break
            print("Image received : {}".format(image_count))
        image_count = image_count + 1
        file.close()
    except:
        print("Error receiving image file. Exiting.")
        file.close()
        #break

# image_hm = cv.imread("server.jpg", 0)
# heatmap = cv.applyColorMap(image_hm, cv.COLORMAP_WINTER)
# cv.imshow('heatmap', heatmap)

img = mpimg.imread('server_{}.jpg'.format(max_img_count - 1))
plt.imshow(img, cmap='winter')
plt.show()

client_socket.close()
