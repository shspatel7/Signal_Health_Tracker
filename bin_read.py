import sys
import socket
import struct
import numpy as np
import matplotlib.pyplot as plt

BYTES_PER_FLOAT = 4 # All data is currently returned as array of floats

def run_getData_saveToFile(decimatorSocket, arg1, arg2, filename):
    decimatorSocket.send('getData:{},{}\r\n'.format(arg1, arg2).encode('utf-8'))
    getData_buf = decimatorSocket.recv(128)

    if getData_buf.startswith(b'getData'):
        # Get the size of data block to follow.
        #  Response looks like  getData:iii,qqq,bbb,\r\n<binary_data>
        #  bbb is the blockSize we need
        rec = getData_buf.split(b',',3)
        blockSize = int(rec[2]) / BYTES_PER_FLOAT
        BUF_SIZE = int(rec[2])

        #determine length of headers
        tmp = rec[3]
        tmp = tmp[2:] # discard \r\n from beginning of block
        headers_len = len(getData_buf) - len(tmp)

        print('looking for {} of data, headers are {}'.format(BUF_SIZE, headers_len))

        print('length is {} of {}'.format(len(getData_buf), BUF_SIZE + headers_len))
        # retry until we have a complete block of data
        while len(getData_buf) < BUF_SIZE + headers_len:
            newdata = decimatorSocket.recv(BUF_SIZE + headers_len - len(getData_buf))
            getData_buf = getData_buf + newdata
            print('length is {} of {}, added {}'.format(len(getData_buf), BUF_SIZE + headers_len, len(newdata)))

        # write to file
        write_file = open(filename, 'wb')
        write_file.write(getData_buf)
        write_file.flush()
        write_file.close()

        # Read back from file
        read_file = open(filename, 'rb')
        read_data = read_file.read()

        # extract binary data from file data
        # unpack is setup to do endian conversion if necessary
        bindata = struct.unpack('!%df' % blockSize, read_data[headers_len:])
        return bindata
    else:
        print("Error: received {}".format(getData_buf))
        sys.exit(1)

def connectToDecimator(ip):
    decimatorSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    decimatorSocket.connect((ip, 9784))
    decimatorSocket.settimeout(10)
    connectBuf = decimatorSocket.recv(128).decode('utf-8')
    if connectBuf.find('connected') == -1:
        print('Cannot connect to Decimator: {}'.format(connectBuf))
        sys.exit(1)
    else:
        print('Connected to decimator.')
    return decimatorSocket

if __name__ == "__main__":

    decimatorSocket = connectToDecimator('192.168.29.3')
    decimatorSocket.send('switchPort:8\r\n'.encode('utf-8'))
    _ = decimatorSocket.recv(100)
    decimatorSocket.send('configSpectrum:1000000000,4000000,200000,Blackman-Harris,0,5,400\r\n'.encode('utf-8'))

    configSpectrum_buf = decimatorSocket.recv(100)


    if not configSpectrum_buf.startswith(b'configSpectrum'):
        msg = b'Error configuring Decimator:' + configSpectrum_buf
        print(msg)
        sys.exit(1)

    bindata = run_getData_saveToFile(decimatorSocket, 1, 1, 'getData_freq_capture.bin')

    plt.plot(bindata)
    plt.title('Frequency capture')
    plt.show()


    decimatorSocket.send('configTime:1001000000,4000000,1,400\r\n'.encode('utf-8'))
    configTime_buf = decimatorSocket.recv(100)

    if not configTime_buf.startswith(b'configTime'):
        msg = 'Error configuring Decimator:' + configTime_buf
        print(msg)
        sys.exit(1)

    bindata = run_getData_saveToFile(decimatorSocket, 2, 0, 'getData_time_capture.bin')

    plt.plot(bindata[0::2])
    plt.plot(bindata[1::2])
    plt.title('Time capture')
    plt.show()