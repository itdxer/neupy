import os
import re
import shutil
import tarfile
from itertools import product

import requests
import pandas as pd
from tqdm import tqdm

from src.utils import create_logger, DATA_DIR, REVIEWS_FILE


logger = create_logger(__name__)

ARCHIVE_URL = ('http://ai.stanford.edu/~amaas/data/'
               'sentiment/aclImdb_v1.tar.gz')
EXTRACTED_DIRECTORY = os.path.join(DATA_DIR, 'aclImdb')
REVIEW_DATA_PATH = os.path.join(DATA_DIR, 'reviews')


def download_file(url, filepath):
    response = requests.get(url, stream=True)
    logger.info("Downloading {}".format(url))

    with open(filepath, "wb") as local_file:
        local_file.write(response.content)

    logger.info('File downloaded and saved succesfully')


def remove_tags_from_text(text):
    tags_regexp = re.compile(r'<[^>]+>')
    return tags_regexp.sub('', text)


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    archive_name = os.path.basename(ARCHIVE_URL)
    path_to_archive = os.path.join(DATA_DIR, archive_name)

    if not os.path.exists(path_to_archive):
        download_file(ARCHIVE_URL, path_to_archive)
    else:
        logger.info("Archive {} has already downloaded".format(archive_name))

    if not os.path.exists(REVIEW_DATA_PATH):
        logger.info("Extracting files from the archive")

        with tarfile.open(path_to_archive) as tar_archive_file:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_archive_file, path=DATA_DIR)

        shutil.move(EXTRACTED_DIRECTORY, REVIEW_DATA_PATH)
    else:
        logger.info("Files have been already extracted.")

    dataset_types = ['train', 'test']
    sentiment_types = ['pos', 'neg']
    data = []

    for dataset_type, review in product(dataset_types, sentiment_types):
        datapath = os.path.join(REVIEW_DATA_PATH, dataset_type, review)

        desc = 'read {:<5s} and {}'.format(dataset_type, review)
        for filename in tqdm(os.listdir(datapath), leave=True, desc=desc):
            filepath = os.path.join(datapath, filename)

            with open(filepath, 'r') as f:
                raw_text = f.read()
                text = remove_tags_from_text(raw_text)
                data.append((text, dataset_type, review))

    dataset = pd.DataFrame(data, columns=['text', 'type', 'sentiment'])
    logger.info("Saving data in CSV file")
    dataset.to_csv(REVIEWS_FILE, sep='\t', index=False)
