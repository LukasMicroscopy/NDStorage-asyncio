from cpython.collections cimport deque  # Import Cython deque
from libc.stdio cimport FILE, fopen, fclose, fwrite
from libc.stdlib cimport malloc, free
from cpython.bytes cimport PyBytes_AsStringAndSize
from cpython.unicode cimport PyUnicode_AsUTF8

cimport cython
cimport numpy as cnp

import json

cdef class SimpleFile:
    cdef FILE* file
    cdef deque write_queue  # Writing queue

    def __init__(self, str filename):
        cdef const char* c_filename
        c_filename = filename.encode('utf-8')
        # Open the file in write mode
        self.file = fopen(c_filename, "w+")
        if not self.file:
            raise IOError("Could not open file for writing")
        # Initialize the deque
        self.write_queue = deque()

    def __dealloc__(self):
        if self.file:
            fclose(self.file)

    def write_numpy_array(self, cnp.ndarray data):
        # Enqueue the numpy array for writing
        self.write_queue.append(data)

    def write_bytearray(self, bytearray data):
        # Enqueue the bytearray for writing
        self.write_queue.append(data)

    def write_dict_as_json(self, dict data):
        # Serialize the dictionary to a JSON string and enqueue it
        json_bytes = json.dumps(data).encode('utf-8')
        self.write_queue.append(json_bytes)

    def process_queue(self):
        """
        Process the write queue and write all enqueued data to the file.
        """
        cdef object data
        cdef const unsigned char* readonly_data
        cdef size_t size

        while self.write_queue:
            # Dequeue the next item
            data = self.write_queue.popleft()

            # Determine the size and data pointer
            if isinstance(data, (bytes, bytearray)):
                readonly_data = <const unsigned char*>data
                size = len(data)
            elif isinstance(data, cnp.ndarray):
                readonly_data = <const unsigned char*>data.data
                size = data.size * data.itemsize
            else:
                raise TypeError("Unsupported data type")

            # Write the data to the file
            fwrite(<const void*>readonly_data, 1, size, self.file)