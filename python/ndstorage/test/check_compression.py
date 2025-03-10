# check how the zlib compression works in terms of compression ratio and data types
import numpy as np
import zlib

def test_compression_for_random_np_array():
    image_height = 2000
    image_width = 2000
    # create a random numpy array
    np.random.seed(0)
    # generate a random array of 2000x2000 elements
    random_array = np.random.randint(low=10,high=200,size=(image_height,image_width),dtype=np.uint16)
    # save the array to a temporary file
    print(f"Original array size: {random_array.nbytes}")
    print(f"Original array type: {random_array.dtype}")
    print(f"Original dimensions: {random_array.shape}")
    print(f"Original array length: {len(random_array.tobytes())}")
    compressed_array = zlib.compress(random_array)
    print(f"Compressed array size: {len(compressed_array)}")
    # check the compression ratio
    print(f"Compression ratio: {random_array.nbytes / len(compressed_array)}")
    # check the data type of the compressed array
    print(f"Compressed array type: {type(compressed_array)}")
    print(f"Compressed array length: {len(compressed_array)}")
    # decompress the array
    decompressed_array = zlib.decompress(compressed_array)
    print(f"Decompressed array size: {len(decompressed_array)}")
    print(f"Decompressed array type: {type(decompressed_array)}")
    # check if the arrays are equal
    assert np.all(random_array.tobytes() == decompressed_array)
    print("byte arrays are equal")
    # create numpy array from decompressed bytes
    decompressed_array = np.frombuffer(decompressed_array, dtype=np.uint16).reshape((image_height, image_width))
    print(f"Decompressed array type: {decompressed_array.dtype}")
    print(f"Decompressed dimensions: {decompressed_array.shape}")
    assert np.all(random_array == decompressed_array)
    print("Arrays are equal")

def test_compression_for_one_np_array():
    image_height = 2000
    image_width = 2000
    time_counter = 123
    array = np.ones(image_height * image_width, dtype=np.uint8).reshape((image_height, image_width)) * time_counter
    print(f"Original array size: {array.nbytes}")
    print(f"Original array type: {array.dtype}")
    compressed_array = zlib.compress(array)
    print(f"Compressed array size: {len(compressed_array)}")
    # check the compression ratio
    print(f"Compression ratio: {array.nbytes / len(compressed_array)}")
    # check the data type of the compressed array
    print(f"Compressed array type: {type(compressed_array)}")
    # decompress the array
    decompressed_array = zlib.decompress(compressed_array)
    print(f"Decompressed array size: {len(decompressed_array)}")
    print(f"Decompressed array type: {type(decompressed_array)}")
    # check if the arrays are equal
    assert np.all(array.tobytes() == decompressed_array)
    # create numpy array from decompressed bytes
    decompressed_array = np.frombuffer(decompressed_array, dtype=np.uint8).reshape((image_height, image_width))
    assert np.all(array == decompressed_array)
    print("Arrays are equal")

if __name__ == "__main__":
    test_compression_for_random_np_array()
    test_compression_for_one_np_array()