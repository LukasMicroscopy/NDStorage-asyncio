
# cython: language_level=3
# distutils: language=c++

cimport cython
#from cpython.collections cimport deque  # Import Cython deque
from libc.stdio cimport fopen, fclose, fwrite, fread, fseek, ftell, fflush, FILE, SEEK_SET, SEEK_CUR, SEEK_END
from libcpp cimport bool as bool_t

import numpy as np
cimport numpy as cnp
import sys
import json
import os
#import time
import struct
import warnings
import mmap
import zlib
from threading import Lock

#from collections import OrderedDict
#from io import BytesIO
from ndstorage.file_io import NDTiffFileIO, BUILTIN_FILE_IO
from ndstorage.ndtiff_index import NDTiffIndexEntry
from ndstorage.ndtiff_file import SingleNDTiffReader

from collections import deque

#from concurrent.futures import ThreadPoolExecutor

#cdef int MAJOR_VERSION = 3
#cdef int MINOR_VERSION = 3

# Constants for writing files
cdef long int BYTES_PER_GIG = 1073741824
cdef long int MAX_FILE_SIZE = 4 * BYTES_PER_GIG

cdef int ENTRIES_PER_IFD = 13

# Required tags
cdef int WIDTH = 256
cdef int HEIGHT = 257
cdef int BITS_PER_SAMPLE = 258
cdef int COMPRESSION = 259
cdef int PHOTOMETRIC_INTERPRETATION = 262
cdef int IMAGE_DESCRIPTION = 270
cdef int STRIP_OFFSETS = 273
cdef int SAMPLES_PER_PIXEL = 277
cdef int ROWS_PER_STRIP = 278
cdef int STRIP_BYTE_COUNTS = 279
cdef int X_RESOLUTION = 282
cdef int Y_RESOLUTION = 283
cdef int RESOLUTION_UNIT = 296
cdef int MM_METADATA = 51123

cdef int SUMMARY_MD_HEADER = 2355492


cdef const char* _POSITION_AXIS = "position"
cdef const char* _ROW_AXIS = "row"
cdef const char* _COLUMN_AXIS = "column"
cdef const char* _Z_AXIS = "z"
cdef const char* _TIME_AXIS = "time"
cdef const char* _CHANNEL_AXIS = "channel"


cdef class SingleNDTiffWriter:
    # type declarations
    #cdef str filename
    cdef dict index_map
    cdef int next_ifd_offset_location
    cdef int res_numerator
    cdef int res_denominator
    cdef int z_step_um
    cdef readonly str filename
    cdef object buffers # deque
    cdef object write_lock # Lock
    cdef readonly object reader # SingleNDTiffReader
    cdef bool_t first_ifd
    cdef int pixel_compression
    cdef FILE* cfile

    def __init__(self, directory, filename, summary_md, pixel_compression = 1):
        self.filename = os.path.join(directory, filename)
        print(f"Initialize writer {self.filename}")
        self.index_map = {}
        self.next_ifd_offset_location = -1
        self.res_numerator = 1
        self.res_denominator = 1
        self.z_step_um = 1
        self.buffers = deque()
        self.first_ifd = True

        self.write_lock = Lock()

        if pixel_compression in [1, 8]:
            self.pixel_compression = pixel_compression
        else:
            raise ValueError("Invalid pixel compression, only 1 (no compression) and 8 (zlib) are supported")
        
        os.makedirs(directory, exist_ok = True)
        
        # Open the file for writing
        # get file name as byte stream
        cdef char* filename_byte_stream 
        encode_filename = self.filename.encode('utf-8')
        filename_byte_stream = <char*>encode_filename
        
        # "w+": Open for reading and writing.
        self.cfile = fopen(filename_byte_stream, 'w+')
        # make sure to start writing at the beginning of the file
        fseek(self.cfile, 0, SEEK_SET)

        # write the file header
        self._write_mm_header_and_summary_md(summary_md)
        self.reader = SingleNDTiffReader

    def has_space_to_write(self, cnp.ndarray pixels, dict metadata):
        cdef bool_t rgb = pixels.ndim == 3 and pixels.shape[2] == 3
        cdef int md_length = len(metadata)
        cdef int IFD_size = ENTRIES_PER_IFD * 12 + 4 + 16
        cdef int extra_padding = 5000000  # 5 MB extra padding
        cdef int bytes_per_pixels = self._bytes_per_image_pixels(pixels, rgb)

        cdef long int file_size = ftell(self.cfile)

        cdef long int size = md_length + IFD_size + bytes_per_pixels + extra_padding + file_size

        if size >= MAX_FILE_SIZE:
            return False
        return True

    cdef _write_mm_header_and_summary_md(self, dict summary_md):
        cdef bytes summary_md_bytes = self._get_bytes_from_string(json.dumps(summary_md))
        cdef size_t md_length = len(summary_md_bytes)
        cdef bytearray header_buffer = bytearray(28)

        # 8 bytes for file header
        if sys.byteorder == 'big':
            struct.pack_into('>H', header_buffer, 0, 0x4D4D)
        else:
            struct.pack_into('<H', header_buffer, 0, 0x4949)
        struct.pack_into('<H', header_buffer, 2, 42)
        cdef size_t first_ifd_offset = 28 + md_length
        if first_ifd_offset % 2 == 1:
            first_ifd_offset += 1  # Start first IFD on a word
        struct.pack_into('<I', header_buffer, 4, first_ifd_offset)

        # 12 bytes for unique identifier and major version
        struct.pack_into('<III', header_buffer, 8, 483729, 3, 3)

        # 8 bytes for summaryMD header and summary md length
        struct.pack_into('<II', header_buffer, 20, SUMMARY_MD_HEADER, md_length)

        self.buffers.append(header_buffer)
        #self._append_buffer(header_buffer)
        self.buffers.append(summary_md_bytes)
        #self._append_buffer(summary_md_bytes)

        self._write_buffer()

        #for buffer in [header_buffer, summary_md_bytes]:
        #    self.file.write(buffer)

    def _get_bytes_from_string(self, s):
        return s.encode('utf-8')

    cdef void _write_buffer(self):
        """
        Write the buffer to the file
        """
        while self.buffers:
            buffer = self.buffers.popleft()
            #buffer, size = self.buffers.popleft()
            self._write_object(buffer)
            #fwrite(<const void*>buffer, 1, size, self.cfile)
        with cython.nogil:
            fflush(self.cfile)

    cdef void _append_buffer(self, object obj):
        """
        Write the object to the file
        """
        cdef const unsigned char* data
        cdef bytes json_bytes
        cdef size_t size
        cdef bytes data_bytes
        if isinstance(obj, (bytes, bytearray)):
            # make sure the bytearray doesen't change while we are writing
            data = <const unsigned char*>obj
            # Get the size of the bytearray
            size = len(obj)
            #fwrite(<const void*>data, 1, size, self.file)
            self.buffers.append((data, size))
        elif isinstance(obj, cnp.ndarray):
            # Make sure the arraz is not altered while we are writing
            data = <const unsigned char*>obj.data
            # Get the size of the array in bytes
            size = obj.size * obj.itemsize
            #fwrite(<const void*>data, 1, size, self.file)
            self.buffers.append((data, size))
        elif isinstance(obj, dict):
            # Serialize the dictionary to a JSON string and encode it as UTF-8
            json_bytes = json.dumps(obj).encode('utf-8')
            data = json_bytes
            size = len(json_bytes)
            #fwrite(<const void*>data, 1, size, self.file)
            self.buffers.append((data, size))
        elif isinstance(obj, str):
            # Convert the string to bytes and write it to the file
            data_bytes = obj.encode('utf-8')
            data = <const unsigned char*>data_bytes.data
            # Get the size of the array in bytes
            size = len(data_bytes)
            #fwrite(<const void*>data, 1, size, self.cfile)
            self.buffers.append((data, size))
        else:
            raise TypeError("Unsupported data type")


    cdef void _write_object(self, object obj):
        """
        Write the object to the file
        """
        cdef const unsigned char* data
        cdef bytes json_bytes
        cdef size_t size
        cdef bytes data_bytes
        if isinstance(obj, (bytes, bytearray)):
            # make sure the bytearray doesen't change while we are writing
            data = <const unsigned char*>obj
            # Get the size of the bytearray
            size = len(obj)
            fwrite(<const void*>data, 1, size, self.cfile)
        elif isinstance(obj, cnp.ndarray):
            # Make sure the arraz is not altered while we are writing
            data_bytes = obj.tobytes()
            data = <const unsigned char*>data_bytes
            # Get the size of the array in bytes
            size = obj.size * obj.itemsize
            fwrite(<const void*>data, 1, size, self.cfile)
        elif isinstance(obj, dict):
            # Serialize the dictionary to a JSON string and encode it as UTF-8
            json_bytes = json.dumps(obj).encode('utf-8')
            data = json_bytes
            size = len(json_bytes)
            fwrite(<const void*>data, 1, size, self.cfile)
        elif isinstance(obj, str):
            # Convert the string to bytes and write it to the file
            data_bytes = obj.encode('utf-8')
            data = <const unsigned char*>data_bytes.data
            # Get the size of the array in bytes
            size = len(data_bytes)
            fwrite(<const void*>data, 1, size, self.cfile)
        else:
            raise TypeError("Unsupported data type")

    def finished_writing(self):
        self._write_null_offset_after_last_image()
        #self.cfile.ftruncate()
        fflush(self.cfile)
        # close the file
        fclose(self.cfile)
        self.cfile = NULL

    cdef void _write_null_offset_after_last_image(self):
        cdef bytearray buffer = bytearray(4)
        struct.pack_into('<I', buffer, 0, 0)
        cdef long int current_pos = ftell(self.cfile)
        fseek(self.cfile, self.next_ifd_offset_location, SEEK_SET)
        fwrite(<const void*>buffer, 1, 4,self.cfile)
        fseek(self.cfile, current_pos, SEEK_SET)

    def write_image(self, frozenset index_key, cnp.ndarray pixels, dict metadata, bit_depth='auto', int pixel_compression = 0):
        """
        Write an image to the file

        Parameters
        ----------  
        index_key : frozenset
            The key to index the image
        pixels : np.ndarray or bytearray
            The image data
        metadata : dict or str
            The metadata for the image
        bit_depth : int
            The bit depth of the image

        Returns
        -------
        NDTiffIndexEntry
            The index entry for the image
        """
        if pixel_compression == 0:
            pixel_compression = self.pixel_compression
        
        cdef Py_ssize_t image_height, image_width  # Declare variables for the shape
        image_height, image_width = pixels.shape[0], pixels.shape[1]
        rgb = pixels.ndim == 3 and pixels.shape[2] == 3
        
        if rgb and pixel_compression in [8]:
            warnings.warn(f"Pixel compression {pixel_compression} is not supported for RGB images. Using no compression.")
            pixel_compression = 1
        if not pixel_compression in [1,8]:
            warnings.warn(f"Invalid pixel compression {pixel_compression}: only 1 (no compression) and 8 (zlib) are supported. Using 1 (no compression).")
            pixel_compression = 1
        
        if bit_depth == 'auto':
            bit_depth = 8 if pixels.dtype == np.uint8 else 16
        # if metadata is a dict, serialize it to a json string and make it a utf8 byte buffer
        if isinstance(metadata, dict):
            metadata_bytes = self._get_bytes_from_string(json.dumps(metadata))
        ied = self._write_ifd(index_key, pixels, metadata_bytes, rgb, image_height, image_width, bit_depth, pixel_compression)
        #while self.buffers:
        #    self.file.write(self.buffers.popleft())
        # make sure the file is flushed to disk
        #self.file.flush()
        with self.write_lock:
            self._write_buffer()
        self.index_map[index_key] = ied
        return ied


    cdef _write_ifd(self, frozenset index_key, cnp.ndarray pixels, bytes metadata, bool_t rgb, int image_height, int image_width, int bit_depth, int pixel_compression):
        if ftell(self.cfile) % 2 == 1:
            fseek(self.cfile, 1, SEEK_CUR)  # Make IFD start on word
            #cdef const int zero_byte = 0
            #fwrite(self.cfile, &zero_byte, 1, 1)  # Make IFD start on word
            #self.file.seek(self.file.tell() + 1)  # Make IFD start on word

        cdef int byte_depth = 0
        if isinstance(pixels, bytearray):
            byte_depth = 1
        elif bit_depth == 8:
            byte_depth = 1
        else:
            byte_depth = 2
        
        if pixel_compression == 8:
            compressed_pixels = zlib.compress(pixels)
            bytes_per_image_pixels = len(compressed_pixels)
        else:
            bytes_per_image_pixels = self._bytes_per_image_pixels(pixels, rgb)
        
        cdef int num_entries = 13

        # 2 bytes for number of directory entries, 12 bytes per directory entry, 4 byte offset of next IFD
        # 6 bytes for bits per sample if RGB, 16 bytes for x and y resolution, 1 byte per character of MD string
        # number of bytes for pixels

        cdef int ifd_and_bit_depth_bytes
        ifd_and_bit_depth_bytes = 2 + num_entries * 12 + 4 + (6 if rgb else 0) + 16
        cdef bytearray ifd_and_small_vals_buffer = bytearray(ifd_and_bit_depth_bytes)

        # Needed to reset to zero after last IFD
        self.next_ifd_offset_location = ftell(self.cfile) + 2 + num_entries * 12
        cdef int bits_per_sample_offset = self.next_ifd_offset_location + 4
        cdef int x_resolution_offset = bits_per_sample_offset + (6 if rgb else 0)
        cdef int y_resolution_offset = x_resolution_offset + 8
        cdef int pixel_data_offset = y_resolution_offset + 8
        cdef long metadata_offset = pixel_data_offset + bytes_per_image_pixels

        cdef int next_ifd_offset = metadata_offset + len(metadata)
        if next_ifd_offset % 2 == 1:
            next_ifd_offset += 1  # Make IFD start on word

        cdef int buffer_position = 0
        struct.pack_into('<H', ifd_and_small_vals_buffer, buffer_position, num_entries)
        buffer_position += 2

        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, WIDTH, 4, 1, image_width, buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, HEIGHT, 4, 1, image_height, buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, BITS_PER_SAMPLE, 3, 3 if rgb else 1,
                                                 bits_per_sample_offset if rgb else byte_depth * 8, buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, COMPRESSION, 3, 1, pixel_compression, buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, PHOTOMETRIC_INTERPRETATION, 3, 1,
                                                 2 if rgb else 1, buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, STRIP_OFFSETS, 4, 1, pixel_data_offset,
                                                 buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, SAMPLES_PER_PIXEL, 3, 1, 3 if rgb else 1,
                                                 buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, ROWS_PER_STRIP, 3, 1, image_height,
                                                 buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, STRIP_BYTE_COUNTS, 4, 1,
                                                 bytes_per_image_pixels, buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, X_RESOLUTION, 5, 1, x_resolution_offset,
                                                 buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, Y_RESOLUTION, 5, 1, y_resolution_offset,
                                                 buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, RESOLUTION_UNIT, 3, 1, 3, buffer_position)
        buffer_position += self._write_ifd_entry(ifd_and_small_vals_buffer, MM_METADATA, 2, len(metadata),
                                                 metadata_offset, buffer_position)

        struct.pack_into('<I', ifd_and_small_vals_buffer, buffer_position, next_ifd_offset)
        buffer_position += 4

        if rgb:
            struct.pack_into('<HHH', ifd_and_small_vals_buffer, buffer_position, byte_depth * 8, byte_depth * 8,
                             byte_depth * 8)
            buffer_position += 6

        struct.pack_into('<II', ifd_and_small_vals_buffer, buffer_position, self.res_numerator, self.res_denominator)
        buffer_position += 8
        struct.pack_into('<II', ifd_and_small_vals_buffer, buffer_position, self.res_numerator, self.res_denominator)
        buffer_position += 8

        self.buffers.append(ifd_and_small_vals_buffer)
        #self._append_buffer(ifd_and_small_vals_buffer)
        if pixel_compression in [8]:
            self.buffers.append(compressed_pixels)
            #self._append_buffer(compressed_pixels)
        else:
            self.buffers.append(self._get_pixel_buffer(pixels, rgb))
            #self._append_buffer(self._get_pixel_buffer(pixels, rgb))

        self.buffers.append(metadata)
        #self._append_buffer(metadata)

        self.first_ifd = False

        # Return structured data for putting into the index entry
        pixel_type = {
            8: NDTiffIndexEntry.EIGHT_BIT,
            10: NDTiffIndexEntry.TEN_BIT,
            12: NDTiffIndexEntry.TWELVE_BIT,
            14: NDTiffIndexEntry.FOURTEEN_BIT,
            16: NDTiffIndexEntry.SIXTEEN_BIT,
            11: NDTiffIndexEntry.ELEVEN_BIT
        }.get(bit_depth, NDTiffIndexEntry.EIGHT_BIT_RGB if rgb else None)

        return NDTiffIndexEntry(index_key, pixel_type, pixel_data_offset, image_width, image_height, metadata_offset,
                                len(metadata), self.filename.split(os.sep)[-1], pixel_compression)

    cdef int _write_ifd_entry(self, bytearray buffer, long tag, long dtype, long count, long value, long buffer_position):
        struct.pack_into('<HHII', buffer, buffer_position, tag, dtype, count, value)
        return 12

    def _get_pixel_buffer(self, pixels, rgb):
        if rgb:
            original_pix = pixels
            rgb_pix = bytearray(len(original_pix) * 3 // 4)
            num_pix = len(original_pix) // 4
            for i in range(num_pix):
                rgb_pix[i * 3] = original_pix[i * 4 + 2]
                rgb_pix[i * 3 + 1] = original_pix[i * 4 + 1]
                rgb_pix[i * 3 + 2] = original_pix[i * 4]
            return rgb_pix
        else:
            return pixels

    cdef int _bytes_per_image_pixels(self, cnp.ndarray pixels, bool_t rgb):
        if rgb:
            return len(pixels) * 3 // 4
        else:
            if isinstance(pixels, bytearray):
                return len(pixels)
            elif isinstance(pixels, cnp.ndarray) and pixels.dtype == np.uint16:
                return pixels.size * 2
            elif isinstance(pixels, cnp.ndarray) and pixels.dtype == np.uint8:
                return pixels.size
            else:
                raise RuntimeError("unknown pixel type")
