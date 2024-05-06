'''
This script contains the code to create a socket server on the host PC to receive the results of the stereo vision.
The scripts uses socket connection and acts as a server for the connection.
It also displays the resultant depth map image in a colourmap using the pyplot.

Please refer the instructions mentioned here -
https://github.com/cu-ecen-aeld/buildroot-assignments-base/wiki/OpenCV-Stereo-Vision
'''
import socket
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import ImageFile

# Specifyies that the received image might be truccated and avoids the exception
ImageFile.LOAD_TRUNCATED_IMAGES = True

image_count = 0
max_img_count = 3

# Creating a socket connection
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # AF_INET = IP, SOCK_STREAM = TCP
# Setting the port to be reusable
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Binding the port for a socket connection.
# Replace this IP address in accordance to the IP of your host PC
server.bind(('10.0.0.46', 1002))

server.listen()
print("Socket server opened. Waiting for connection..")

# Save the client details once the connection is established
client_socket, client_address = server.accept()

print("Connected to client at address {}".format(client_address))

# Setting a timeout to receive images from client(raspi)
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

# Display the depth image in a colourmap format
# You can change the cmap parameter for displaying different patterns for ease in visual analysis
img = mpimg.imread('server_{}.jpg'.format(max_img_count - 1))
plt.imshow(img, cmap='winter')
plt.show()

# Closing the connection
client_socket.close()
