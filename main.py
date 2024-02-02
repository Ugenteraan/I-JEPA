'''Main module for this repo.
'''


import argparse


from train import train



def main():


    #write logics to create necessary folders automatically later.

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Specify the YAML config file to be used.')
    parser.add_argument('--logging_config', required=True, type=str, help='Specify the YAML config file to be used for the logging module.')
    parser.add_argument('--task', required=True, type=str, help='Specify whether the purpose is to train the I-JEPA model or to use it to downstream. [train, downstream]')

    args = parser.parse_args()



    




if __name__ == '__main__':

    main()
