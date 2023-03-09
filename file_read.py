import sys
import struct
import matplotlib.pyplot as plt
import numpy as np
import time

BYTES = 4

f = "getData_freq_capture.bin"

#Read the bin file and get data out of it
with open(f, 'rb') as f:
    data = f.read()
    if data.startswith(b'getData'):
        print("File is OK")
        rec = data.split(b',',3)
        blockSize = int(rec[2]) / BYTES
        buf_size = int(rec[2])
        print('blockSize is {}'.format(blockSize))
        tmp = rec[3]
        tmp = tmp[2:] # discard \r\n from beginning of block
        headers_len = len(data) - len(tmp)
        print('looking for {} of data, headers are {}'.format(buf_size, headers_len))


#Write the data to a new file
with open("new.bin", 'wb') as f:
    f.write(data)
    f.flush()
    f.close()

#Read the data from the new file
with open("new.bin", 'rb') as f:
    data = f.read()
    # extract binary data from file data
    # unpack is setup to do endian conversion if necessary
    bindata = struct.unpack('!%df' % blockSize, data[headers_len:])
    # I | Q data where the every pair is I and Q data in the stream
    # I data is in even index and Q data is in odd index
    print(bindata)
    # Calculate Statistics
    print("Mean: {}".format(np.mean(bindata)))
    print("Std: {}".format(np.std(bindata)))
    print("Max: {}".format(np.max(bindata)))
    print("Min: {}".format(np.min(bindata)))
    print("Median: {}".format(np.median(bindata)))
    print("Variance: {}".format(np.var(bindata)))

    # plt.plot(bindata[0:100])
    # plt.show()
    # I_data = bindata[0::2]
    # Q_data = bindata[1::2]
    #
    # plt.plot(I_data[100:120])
    # plt.plot(Q_data[100:120])
    # plt.show()
    #Power = I^2 + Q^2
    # power = [x**2 + y**2 for x, y in zip(I_data, Q_data)]
    # print(power)
    # Convert power to dBm
    # power = [10 * np.log10(x) for x in bindata]
    # print(power)
    #write power to a csv file along with counter
    with open("new.csv", 'w') as f:
        for i in range(len(bindata)):
            f.write("{},{}\r".format(i, bindata[i]))

    plt.plot(bindata[100:200])
    plt.show()