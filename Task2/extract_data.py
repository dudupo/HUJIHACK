import os


def extract_data_to_files(project_name, path='PATH_TO_PROJECTS'):
    datalist = []
    all_data_file = open('PATH_TO_PROJECTS' + project_name + '_all_data.txt', 'w')
    for path, subdirs, files in os.walk(os.path.join(path, project_name + '-master')):
        for name in files:
            _, file_extension = os.path.splitext(name)
            if (file_extension in ['.py', '.js', '.go', '.h', '.cc', '.java', '.dart']):
                file = open(os.path.join(path, name), 'r')
                dataFile = file.readlines()
                for line in dataFile:
                    all_data_file.write(line)
                # datalist.extend(dataFile)
                file.close()
    all_data_file.close()


if __name__ == '__main__':
    project_list = ['building_tool', 'espnet', 'horovod', 'jina', 'PaddleHub', 'PySolFC', 'pytorch_geometric']
    for p in project_list:
        extract_data_to_files (p)
