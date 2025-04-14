from simple_file import SimpleFile
import numpy as np



def test_simple_file():
    file_name = "/media/zimuser/Daten/test_simple_file.dat"
    # Create a SimpleFile object
    sf = SimpleFile(file_name)

    # Write some data to the file
    data = np.random.rand(10, 10).astype(np.float32)
    sf.write_numpy_array(data)
    
    # Write dictionary data to the file
    dict_data = {
        "key1": 'FUCK',
        "key2": 42
    }
    sf.write_dict_as_json(dict_data)
    
    sf = None
    
    # Read the data back from the file
    reader = file.open(file_name, mode='rb')
    data_read = np.fromfile(reader, dtype=np.float32, count=100).reshape(10, 10)
    assert np.array_equal(data, data_read)
    # Read the dictionary data back from the file
    # reader.seek(0, 2)  # Move to the end of the file
    # dict_data_read = reader.read()
    # Assuming the dictionary data is stored in a specific format,
    # you would need to parse it accordingly
