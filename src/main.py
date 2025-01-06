import config  # For setting the root directory of the project.
from config import SUBMISSION_PATH

from typing import List
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import pandas as pd
from src.model.base_model import BaseModel
from src.model.spine_cnn import MultiModelSpineCNN
from src.dataset.spine_dataset import SingleModelLumbarSpineDataset, MultiModelLumbarSpineDataset
from src.dataset.augmentations import add_gaussian_noise


def make_submission_with_loaded_multi_model(model: MultiModelSpineCNN, save_suffix: str) -> None:
    output_df = pd.DataFrame(columns=["row_id", "normal_mild", "moderate", "severe"])
    test_dataset = MultiModelLumbarSpineDataset(False)
    dir_path = SUBMISSION_PATH / f"{model.name}_submissions"
    dir_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            test_dict = test_dataset[idx]
            test_dict["data"] = [torch.unsqueeze(data, 0) for data in test_dict["data"]]
            y_hat = model(test_dict).view(25, 3)  # Only one sample in the test set.
            y_hat = F.softmax(y_hat, dim=1)
            curr_df = pd.DataFrame(
                {
                    "row_id": [f"{test_dict['row_id']}_{condition}" for condition in test_dataset.conditions_i2s],
                    "normal_mild": y_hat[:, 0].numpy(),
                    "moderate": y_hat[:, 1].numpy(),
                    "severe": y_hat[:, 2].numpy()
                }
            )
            output_df = pd.concat([output_df, curr_df], axis=0, ignore_index=True)
    output_df.sort_values(by="row_id", inplace=True)
    output_df.to_csv(dir_path / f"{model.name}_{save_suffix}.csv", index=False)


def make_submission(*model_init_dicts, epoch: int, **model_init_kwargs) -> None:
    model = MultiModelSpineCNN(*model_init_dicts, **model_init_kwargs)
    model.load_state_dict(torch.load(model.model_dir / f"{model.name}_e={epoch}.pt"))
    make_submission_with_loaded_multi_model(model, f"e={epoch}")


def main() -> None:

    # Was better without augmentation.
    dataset = MultiModelLumbarSpineDataset(train=True)#, augs=add_gaussian_noise)
    train_dataset, val_dataset = dataset.split(val_size=0.15)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=3)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=3)

    # sag_t1_args = dict(architecture="MC3_18", pretrained=True, progress=True, out_features_size=512)
    sags_args = dict(architecture="R3D_18", pretrained=False, progress=True, out_features_size=1024)
    axial_t2_args = dict(architecture="R3D_18", pretrained=False, progress=True, out_features_size=750)
    model = MultiModelSpineCNN(sags_args, axial_t2_args, last_fc_dim=1024 + 750, dropout=0.4,
                               name="multi_model_submission_version")
    model.fit(train_loader=train_dataloader, val_loader=val_dataloader, num_epochs=15,
              lr=0.001, wd=0.0005, try_cuda=True, verbose=True, print_stride=1)
    make_submission_with_loaded_multi_model(model, "final")


if __name__ == '__main__':
    main()
