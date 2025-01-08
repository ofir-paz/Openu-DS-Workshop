import sys
import os
from pathlib import Path


if __name__ == "__main__":
    # Add the root directory to the Python path
    PROJECT_PATH = Path(os.path.dirname(__file__)).parent
    sys.path.append(str(PROJECT_PATH))

    print(f"Added {PROJECT_PATH} to the Python path.")

    from torch.utils.data import DataLoader
    from src.model.spine_cnn import MultiModelSpineCNN
    from src.dataset.spine_dataset import SingleModelLumbarSpineDataset, MultiModelLumbarSpineDataset

    dataset = MultiModelLumbarSpineDataset(train=True)
    train_dataset, val_dataset = dataset.split(val_size=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    sag_t1_args = dict(architecture="MC3_18", pretrained=True, progress=True, out_features_size=512, inst_name="sag_t1")
    sag_t2_args = dict(architecture="MC3_18", pretrained=True, progress=True, out_features_size=512, inst_name="sag_t2")
    axial_t2_args = dict(architecture="MC3_18", pretrained=True, progress=True, out_features_size=1024, inst_name="axial_t2")

    model = MultiModelSpineCNN(sag_t1_args, sag_t2_args, axial_t2_args, last_fc_dim=1024, dropout=0.5,
                               name="multi_model_v2")
    model.fit(train_loader=train_dataloader, val_loader=val_dataloader, num_epochs=30,
              lr=0.0025, momentum=0.9, wd=0.0005, try_cuda=True, verbose=True, print_stride=1)
