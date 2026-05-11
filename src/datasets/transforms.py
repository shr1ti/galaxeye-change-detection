import numpy as np

try:
    import albumentations as A
except ModuleNotFoundError:
    A = None


class NumpyTrainTransform:
    def __call__(self, image, mask):
        if np.random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        if np.random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        if np.random.random() < 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k=k, axes=(0, 1))
            mask = np.rot90(mask, k=k, axes=(0, 1))

        return {
            "image": np.ascontiguousarray(image),
            "mask": np.ascontiguousarray(mask),
        }


class NumpyValTransform:
    def __call__(self, image, mask):
        return {"image": image, "mask": mask}


if A is not None:
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
    val_transform = A.Compose([])
else:
    train_transform = NumpyTrainTransform()
    val_transform = NumpyValTransform()
