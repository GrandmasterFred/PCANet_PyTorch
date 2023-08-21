import os
import pickle
import shutil
import numpy as np
import logging


def create_exp_dir(path, scripts_to_save=None):
    """
    copy executed python files to the path
    :param path: save path
    :param scripts_to_save: a list of executed python files 
    :return: 
    """
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
        dst_file = os.path.join(path, 'scripts', os.path.basename(script))
        shutil.copyfile(script, dst_file)


def save_feature(batch_feature, filename):
    with open(filename, "ab+") as f:
        pickle.dump(batch_feature, f)


def create_pickle_file_name(stage_save_path, stage):
    dirname = os.path.join(stage_save_path, "stage_" + str(stage))
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    filename = os.path.join(dirname, "stage_" + str(stage) + "_feature.pkl")

    return filename


def load_feature(filename, old_pointer):
    """
    read features and update file object pointer
    :param filename: 
    :param old_pointer: 
    :return: 
    """
    with open(filename, "rb") as f:
        f.seek(old_pointer, 0)
        features = pickle.load(f)
        new_pointer = f.tell()
    return features[0], features[1], new_pointer


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def total_accuracy(predict_class, test_labels):
    temp = np.equal(predict_class, test_labels)
    num_of_correct_samples = temp.sum().item()
    return num_of_correct_samples


def exchange_channel(single_feature):
    """
    It is convenient to display feature map in tensorboardX using this function 
    :param single_feature: 
    :return: 
    """
    single_feature = single_feature.unsqueeze(dim=1)
    return single_feature

# make a class that has the logger, as well as the method of printing i guess

class MyLogger:
    def __init__(self, log_file, log_level=logging.DEBUG):
        # Configure logging settings
        self.log_file = log_file
        self.log_level = log_level
        self._configure_logging()

    def _configure_logging(self):
        logging.basicConfig(filename=self.log_file, level=self.log_level,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def log(self, message, level=logging.INFO):
        if level == logging.DEBUG:
            logging.debug(message)
        elif level == logging.INFO:
            logging.info(message)
        elif level == logging.WARNING:
            logging.warning(message)
        elif level == logging.ERROR:
            logging.error(message)
        elif level == logging.CRITICAL:
            logging.critical(message)
        else:
            raise ValueError("Invalid log level")
        # this then also prints to console as well
        print(message)


if __name__ == "__main__":
    filename = create_pickle_file_name(0)
    print(filename)

    import torch
    for i in range(1, 4):
        batch_feature = torch.arange(24).view(-1, i, 2, 2).float()
        save_feature(batch_feature, filename)
        print("save completely")

    features = load_feature(filename)
    print("load completely")
