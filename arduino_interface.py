import socket
import time
import keyboard
from live_capture_ocr import ocr_location
from live_capture_bolt import bolt_location

sock = socket.socket()

host = "192.168.43.181"  # ESP32 IP in local network
port = 80  # ESP32 Server Port

sock.connect((host, port))
data1 = 1027
data2 = 2000
data3 = 6969
#message = "{},{}".format(data1,data2)
# sock.send(message.encode("utf-8"))


while True:
    print(keyboard.read_key())
    if keyboard.read_key() == "o":
        data2, data3 = ocr_location()
    else:
        data2, data3 = bolt_location()
    data1 += 1
    message = "{},{},{},&".format(data1, data2, data3)
    sock.send(message.encode("utf-8"))
    time.sleep(1)
#data = ""

# while len(data) < len(message):
#    data += sock.recv(1)

# print(data)

# sock.close()
