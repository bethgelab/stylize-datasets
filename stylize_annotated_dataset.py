from Datasets import StylizeDataset, get_dataset_from_config
import argparse


parser = argparse.ArgumentParser(description='This script applies the AdaIN style transfer method to arbitrary annotated datasets.')

parser.add_argument('--config-path', type=str, help='Path to the dataset config file')

def main():
    args = parser.parse_args()

    dataset = get_dataset_from_confg(args.config_path)
    print('Copying the dataset from %s to %s...')
    dataset.copy()
    print('Finished!')
    print('Stylizing the dataset in %s')
    dataset.stylize()



if __name__ == '__main__':
    main()
