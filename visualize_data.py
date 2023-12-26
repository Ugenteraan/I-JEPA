'''Module to visualize the dataset.
'''


from load_dataset import DeepLakeDataset
import matplotlib.pyplot as plt




class VisualizeData:

    def __init__(self, deeplake_module, deeplake_dataset, visualize_batch_size, visualize_shuffle, deeplake_token):

        self.dataloader = deeplake_module(token=deeplake_token, deeplake_dataset=deeplake_dataset, batch_size=visualize_batch_size, shuffle=visualize_shuffle)

    def __call__(self):

        
         


m = DeepLakeDataset(token=cred.DEEPLAKE_TOKEN, deeplake_dataset='hub://activeloop/imagenet-train', batch_size=2, shuffle=False)()
