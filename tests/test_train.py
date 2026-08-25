import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from modelSS import modelSS
from modelSS_train import (
    SegmentationTransform,
    evaluate,
    iou_from_confusion_matrix,
    mean_iou,
    train,
    update_confusion_matrix,
)


class FakeSegDataset(Dataset):
    """Tiny synthetic stand-in for VOCSegmentation (64x64, a few void pixels)."""

    def __init__(self, n=4, size=64, num_classes=21):
        self.n = n
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        image = torch.randn(3, self.size, self.size, generator=g)
        target = torch.randint(0, self.num_classes, (self.size, self.size), generator=g)
        target[0, :] = 255  # a row of void pixels
        return image, target


def test_confusion_matrix_excludes_void():
    nc = 3
    target = torch.tensor([[[0, 1], [255, 2]]])
    outputs = torch.zeros(1, nc, 2, 2)
    outputs[0, 0, 0, 0] = 10
    outputs[0, 1, 0, 1] = 10
    outputs[0, 0, 1, 0] = 10  # sits on the void pixel, must be ignored
    outputs[0, 2, 1, 1] = 10

    cm = update_confusion_matrix(torch.zeros(nc, nc, dtype=torch.int64), outputs, target, nc)

    assert cm.sum().item() == 3  # 4 pixels minus 1 void
    assert mean_iou(cm) == 1.0


def test_iou_marks_absent_classes_nan():
    nc = 3
    target = torch.tensor([[[0, 0], [1, 1]]])
    outputs = torch.zeros(1, nc, 2, 2)
    outputs[0, 0] = 10  # predict class 0 everywhere

    cm = update_confusion_matrix(torch.zeros(nc, nc, dtype=torch.int64), outputs, target, nc)
    ious = iou_from_confusion_matrix(cm)

    assert ious[0].item() == 0.5
    assert ious[1].item() == 0.0
    assert torch.isnan(ious[2])  # class 2 in neither pred nor target


def test_transform_is_picklable():
    import pickle

    t = pickle.loads(pickle.dumps(SegmentationTransform()))
    assert t.resize.size == (256, 256)


def test_evaluate_runs_without_grad():
    device = torch.device('cpu')
    model = modelSS(num_classes=21)
    loader = DataLoader(FakeSegDataset(), batch_size=2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    loss, miou, class_ious = evaluate(model, loader, loss_fn, device)

    assert loss > 0
    assert 0.0 <= miou <= 1.0
    assert class_ious.shape == (21,)


def test_train_loop_runs_and_checkpoints_best(tmp_path):
    device = torch.device('cpu')
    model = modelSS(num_classes=21)
    train_loader = DataLoader(FakeSegDataset(), batch_size=2)
    val_loader = DataLoader(FakeSegDataset(), batch_size=2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    save_file = tmp_path / 'w.pth'
    train(
        n_epochs=2,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=None,
        device=device,
        save_file=str(save_file),
        plot_file=None,
    )

    assert save_file.exists()  # best-val checkpoint was written
