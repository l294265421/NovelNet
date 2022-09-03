import os
import sys


def write(message: str):
    """

    """
    if sys.platform != 'darwin':
        with open('/data/ceph/yuncongli/fiction-recommendation/session-rec-pytorch-private/log.txt', mode='a') \
            as output_file:
            output_file.write(str(message))
            output_file.write(os.linesep)
            output_file.flush()
