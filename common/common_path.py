import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.platform != 'darwin':
    project_dir = '/data/ceph/yuncongli/fiction-recommendation/session-rec-pytorch-private/'

repeat_aware_rec_base_dir = os.path.join(project_dir, 'rar_base_dir')
external_data_dir = os.path.join(project_dir, 'external_data')

if __name__ == '__main__':
    print(project_dir)
