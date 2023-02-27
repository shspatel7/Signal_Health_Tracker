import sys
import struct
import matplotlib.pyplot as plt

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
            print(bindata)
            plt.plot(bindata)
            plt.show()



