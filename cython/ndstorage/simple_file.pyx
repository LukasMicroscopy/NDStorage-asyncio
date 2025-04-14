# simple_file.pyx
# This Cython module tests writing different Python data structures to a file.

from libc.stdio cimport FILE, fopen, fclose, fprintf, fwrite
from libc.stdlib cimport malloc, free
from cpython.bytes cimport PyBytes_AsStringAndSize
from cpython.unicode cimport PyUnicode_AsUTF8

cimport numpy as cnp

import json

cdef class SimpleFile:
    cdef FILE* file

    def __init__(self, str filename):
        cdef const char* c_filename
        byte_name = filename.encode('utf-8')
        c_filename = byte_name
        # Open the file in write mode
        print("Opening file")
        self.file = fopen(c_filename, "w+")
        if not self.file:
            raise IOError("Could not open file for writing")

    def __dealloc__(self):
        if self.file:
            print("Closing file")
            fclose(self.file)

    def write_numpy_array(self, cnp.ndarray data):
        # Make sure the arraz is not altered while we are writing
        cdef const unsigned char* readonly_data = <const unsigned char*>data.data
        # Get the size of the array in bytes
        cdef size_t size = data.size * data.itemsize
        # Write the array's data to the file
        self.write_bytes(readonly_data, size)

    def write_bytearray(self, bytearray data):
        # make sure the bytearray doesen't change while we are writing
        cdef const unsigned char* readonly_data = <const unsigned char*>data
        # Get the size of the bytearray
        cdef size_t size = len(data)
        # Write the bytearray's data to the file
        self.write_bytes(readonly_data, size)

    def write_dict_as_json(self, dict data):
        # Serialize the dictionary to a JSON string and encode it as UTF-8
        cdef bytes json_bytes = json.dumps(data).encode('utf-8')
        cdef const unsigned char* json_data = json_bytes
        cdef size_t json_size = len(json_bytes)
        # Write the JSON data to the file
        self.write_bytes(json_data, json_size)

    cdef void write_bytes(self, const void* data, size_t size):
        # Write the bytes object's data to the file
        fwrite(<const void*>data, 1, size, self.file)

    cdef size_t get_size(self, object obj):
        """
        Get the size of the data in bytes for supported Python objects.
        """
        if isinstance(obj, (bytes, bytearray)):
            return len(obj)
        elif isinstance(obj, cnp.ndarray):
            return obj.size * obj.itemsize
        else:
            raise TypeError("Unsupported data type")
    
    def write_data(self, object data):
        cdef const unsigned char* readonly_data
        cdef size_t size = self.get_size(data)

        if isinstance(data, (bytes, bytearray)):
            readonly_data = <const unsigned char*>data
        elif isinstance(data, cnp.ndarray):
            readonly_data = <const unsigned char*>data.data
        else:
            raise TypeError("Unsupported data type")

        fwrite(<const void*>readonly_data, 1, size, self.file)

