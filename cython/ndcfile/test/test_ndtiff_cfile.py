import numpy as np
import os
import shutil
from ndstorage.ndtiff_file import SingleNDTiffReader
from ndstorage import NDTiffDataset
from ndcfile import SingleNDTiffWriter
#from ..ndtiff_dataset import NDTiffDataset
#from ..ndram_dataset import NDRAMDataset
import pytest

@pytest.fixture(scope="function")
def test_data_path(tmp_path_factory):
    data_path = tmp_path_factory.mktemp('writer_tests')
    for f in os.listdir(data_path):
        os.remove(os.path.join(data_path, f))
    yield str(data_path)
    shutil.rmtree(data_path)

def test_write_single_file(test_data_path):
    """
    Create a single NDTiff file and read it back in
    """
    filename = 'test_write_single_file.tif'
    writer = SingleNDTiffWriter(test_data_path, filename, summary_md={})
    print(f"initalizing writer: {writer.filename}")

    image_height = 256
    image_width = 256
    pixels = np.arange(image_height * image_width, dtype=np.uint16).reshape((image_height, image_width))

    index_key = frozenset({'time': 0}.items())
    index_entry = writer.write_image(index_key, pixels, {})
    writer.finished_writing()

    # read the file back in
    single_reader = SingleNDTiffReader(os.path.join(test_data_path, filename))
    read_pixels = single_reader.read_image(index_entry)
    assert np.all(read_pixels == pixels)
    print("write test done")
    
test_data_path = "/media/zimuser/Daten/test_simple_file"
test_write_single_file(test_data_path)