from datasets import load_dataset_builder,load_dataset,get_dataset_config_names, Dataset
from datasets.dataset_dict import DatasetDict
from typing import Literal

class DatasetLoader():
    r"""
    A dataset class used to load dataset.

    Args:
        dataset_name (str): The name of the dataset.
        split (str()): load a specific split
    """

    def __init__(
        self,
        dataset_name: str,
        split: Literal["train","validation","test"] | None
    ):
        self.dataset_name = dataset_name
        self.split = split,
        super().__init__()

    def builder(self) -> DatasetDict | Dataset:
        r"""
        Initializes and prepares the datasetbuilder.
        """
        return load_dataset_builder(path=self.dataset_name)

    def load_dataset(self)-> DatasetDict | Dataset:
        r"""
        Return the loaded dataset"""
        if self.split == None:
            return load_dataset(path=self.dataset_name)
        else:
            return load_dataset(path=self.dataset_name, split=self.split)
        
    def get_config(self)-> list:
        r"""
        Return the configuration dataset"""
        return get_dataset_config_names(self.dataset_name)
    
# Example usecase
# d = DatasetLoader(dataset_name='rotten_tomatoes',split=None)
# print(d.builder().info.description)