import pandas as pd

from modelTRA import MyDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def main():
    static = pd.read_csv('ufeature.csv').set_index('uid')
    dynamic = pd.read_csv('trajectory.csv').set_index('uid')
    dynamic = dynamic.set_index('time', append=True)
    index = dynamic.index
    feature = dynamic[['chan0','chan1','chan2','chan3']]
    label = dynamic[['conver']]

    index = dynamic.index
    feature = dynamic[['chan0', 'chan1', 'chan2', 'chan3']].values.astype("float32")
    label = dynamic[['conver']].values.astype("float32")

    HData = MyDataset(feature=feature, label=label, index=index)
    data_loader = DataLoader(HData, batch_size=10)

    # max_step = 50
    # for batch in tqdm(HData, total=max_step):
    #     print(batch)

    for batch_idx, data in enumerate(data_loader):
        print(batch_idx, data)


PARAM = {
    'n_epoch': 100,
    'max_step_per_epoch': 20,
    'pattern': 3,
}

if __name__ == "__main__":
    main()