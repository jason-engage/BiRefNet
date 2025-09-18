import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms

from image_proc import preproc
from config import Config
from utils import path_to_image


Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning
config = Config()
_class_labels_TR_sorted = (
    'Airplane, Ant, Antenna, Archery, Axe, BabyCarriage, Bag, BalanceBeam, Balcony, Balloon, Basket, BasketballHoop, Beatle, Bed, Bee, Bench, Bicycle, '
    'BicycleFrame, BicycleStand, Boat, Bonsai, BoomLift, Bridge, BunkBed, Butterfly, Button, Cable, CableLift, Cage, Camcorder, Cannon, Canoe, Car, '
    'CarParkDropArm, Carriage, Cart, Caterpillar, CeilingLamp, Centipede, Chair, Clip, Clock, Clothes, CoatHanger, Comb, ConcretePumpTruck, Crack, Crane, '
    'Cup, DentalChair, Desk, DeskChair, Diagram, DishRack, DoorHandle, Dragonfish, Dragonfly, Drum, Earphone, Easel, ElectricIron, Excavator, Eyeglasses, '
    'Fan, Fence, Fencing, FerrisWheel, FireExtinguisher, Fishing, Flag, FloorLamp, Forklift, GasStation, Gate, Gear, Goal, Golf, GymEquipment, Hammock, '
    'Handcart, Handcraft, Handrail, HangGlider, Harp, Harvester, Headset, Helicopter, Helmet, Hook, HorizontalBar, Hydrovalve, IroningTable, Jewelry, Key, '
    'KidsPlayground, Kitchenware, Kite, Knife, Ladder, LaundryRack, Lightning, Lobster, Locust, Machine, MachineGun, MagazineRack, Mantis, Medal, MemorialArchway, '
    'Microphone, Missile, MobileHolder, Monitor, Mosquito, Motorcycle, MovingTrolley, Mower, MusicPlayer, MusicStand, ObservationTower, Octopus, OilWell, '
    'OlympicLogo, OperatingTable, OutdoorFitnessEquipment, Parachute, Pavilion, Piano, Pipe, PlowHarrow, PoleVault, Punchbag, Rack, Racket, Rifle, Ring, Robot, '
    'RockClimbing, Rope, Sailboat, Satellite, Scaffold, Scale, Scissor, Scooter, Sculpture, Seadragon, Seahorse, Seal, SewingMachine, Ship, Shoe, ShoppingCart, '
    'ShoppingTrolley, Shower, Shrimp, Signboard, Skateboarding, Skeleton, Skiing, Spade, SpeedBoat, Spider, Spoon, Stair, Stand, Stationary, SteeringWheel, '
    'Stethoscope, Stool, Stove, StreetLamp, SweetStand, Swing, Sword, TV, Table, TableChair, TableLamp, TableTennis, Tank, Tapeline, Teapot, Telescope, Tent, '
    'TobaccoPipe, Toy, Tractor, TrafficLight, TrafficSign, Trampoline, TransmissionTower, Tree, Tricycle, TrimmerCover, Tripod, Trombone, Truck, Trumpet, Tuba, '
    'UAV, Umbrella, UnevenBars, UtilityPole, VacuumCleaner, Violin, Wakesurfing, Watch, WaterTower, WateringPot, Well, WellLid, Wheel, Wheelchair, WindTurbine, Windmill, WineGlass, WireWhisk, Yacht'
)
class_labels_TR_sorted = _class_labels_TR_sorted.split(', ')


class MyData(data.Dataset):
    def __init__(self, datasets, data_size, is_train=True):
        # data_size is None when using dynamic_size or data_size is manually set to None (for inference in the original size).
        self.is_train = is_train
        self.data_size = data_size
        self.load_all = config.load_all
        self.device = config.device
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']

        if self.is_train and config.auxiliary_classification:
            self.cls_name2id = {_name: _id for _id, _name in enumerate(class_labels_TR_sorted)}
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.dataset_names = datasets.split('+')  # Store original order
        self.image_paths = []
        for dataset in self.dataset_names:
            image_root = os.path.join(dataset_root, dataset, 'jpgs')
            self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root) if any(p.endswith(ext) for ext in valid_extensions)]
        self.label_paths = []
        for p in self.image_paths:
            for ext in valid_extensions:
                ## 'jpgs' and 'masks' may need modifying
                p_gt = p.replace('/jpgs/', '/masks/')[:-(len(p.split('.')[-1])+1)] + ext
                file_exists = False
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    file_exists = True
                    break
            if not file_exists:
                print('Not exists:', p_gt)

        if len(self.label_paths) != len(self.image_paths):
            set_image_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.image_paths])
            set_label_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.label_paths])
            print('Path diff:', set_image_paths - set_label_paths)
            raise ValueError(f"There are different numbers of images ({len(self.label_paths)}) and labels ({len(self.image_paths)})")

        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            self.class_labels_loaded = []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = path_to_image(image_path, size=self.data_size, color_type='rgb')
                _label = path_to_image(label_path, size=self.data_size, color_type='gray')
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
                self.class_labels_loaded.append(
                    self.cls_name2id[label_path.split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1
                )

    def __getitem__(self, index):
        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            class_label = self.class_labels_loaded[index] if self.is_train and config.auxiliary_classification else -1
        else:
            image = path_to_image(self.image_paths[index], size=self.data_size, color_type='rgb')
            label = path_to_image(self.label_paths[index], size=self.data_size, color_type='gray')
            class_label = self.cls_name2id[self.label_paths[index].split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1

        # loading image and label
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=config.preproc_methods)
        
		# else:
        #     if _label.shape[0] > 2048 or _label.shape[1] > 2048:
        #         _image = cv2.resize(_image, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        #         _label = cv2.resize(_label, (2048, 2048), interpolation=cv2.INTER_LINEAR)

        # At present, we use fixed sizes in inference, instead of consistent dynamic size with training.
        if self.is_train:
            if config.dynamic_size is None:
                image, label = self.transform_image(image), self.transform_label(label)
        else:
            size_div_32 = (int(image.size[0] // 32 * 32), int(image.size[1] // 32 * 32))
            if image.size != size_div_32:
                image = image.resize(size_div_32)
                label = label.resize(size_div_32)
            image, label = self.transform_image(image), self.transform_label(label)

        if self.is_train:
            return image, label, class_label, self.label_paths[index]
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)

    def get_batch_composition(self, paths):
        """
        Extract dataset names from paths and count occurrences

        Args:
            paths: List of label paths from a batch

        Returns:
            Dictionary with dataset names as keys and counts as values
        """
        counts = {}

        for path in paths:
            # Extract the dataset name from path
            # Path structure: /path/to/data/TASK_NAME/DATASET_NAME/masks/image.jpg
            path_parts = path.split('/')

            # Find the dataset name - it's between the task folder and 'masks'/'jpgs'
            for i, part in enumerate(path_parts):
                if part == 'masks' or part == 'jpgs':
                    if i > 0:
                        dataset_name = path_parts[i - 1]
                        counts[dataset_name] = counts.get(dataset_name, 0) + 1
                        break

        return counts


# Original custom collate function with dynamic size support - Dynamic size changes every batch (VERY SLOW!)
def custom_collate_fn(batch):
    if config.dynamic_size:
        dynamic_size = tuple(sorted(config.dynamic_size))
        dynamic_size_batch = (random.randint(dynamic_size[0][0], dynamic_size[0][1]) // 32 * 32, random.randint(dynamic_size[1][0], dynamic_size[1][1]) // 32 * 32) # select a value randomly in the range of [dynamic_size[0/1][0], dynamic_size[0/1][1]].
        data_size = dynamic_size_batch
    else:
        data_size = config.size
    new_batch = []
    transform_image = transforms.Compose([
        transforms.Resize(data_size[::-1]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_label = transforms.Compose([
        transforms.Resize(data_size[::-1]),
        transforms.ToTensor(),
    ])
    for image, label, class_label, path in batch:
        new_batch.append((transform_image(image), transform_label(label), class_label, path))
    return data._utils.collate.default_collate(new_batch)



# Modified custom collate function with dynamic size support (size changes every N batches)
def custom_collate_resize_fn(batch):
    if config.dynamic_size and config.dynamic_size_batch > 0:
        # Initialize persistent variables if they don't exist
        if not hasattr(custom_collate_resize_fn, 'batch_counter'):
            custom_collate_resize_fn.batch_counter = 0
            custom_collate_resize_fn.current_size = None

        if custom_collate_resize_fn.batch_counter % config.dynamic_size_batch == 0 or custom_collate_resize_fn.current_size is None:
            # Pick new random width and height from the list
            chosen_width = random.choice(config.dynamic_size)
            chosen_height = random.choice(config.dynamic_size)
            custom_collate_resize_fn.current_size = (chosen_width, chosen_height)
            # print(f"[Dynamic Size] Changing to new size: {custom_collate_resize_fn.current_size} at batch {custom_collate_resize_fn.batch_counter}")

        custom_collate_resize_fn.batch_counter += 1
        data_size = custom_collate_resize_fn.current_size
    else:
        data_size = config.size
    new_batch = []
    transform_image = transforms.Compose([
        transforms.Resize(data_size[::-1]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_label = transforms.Compose([
        transforms.Resize(data_size[::-1]),
        transforms.ToTensor(),
    ])
    for image, label, class_label, path in batch:
        new_batch.append((transform_image(image), transform_label(label), class_label, path))
    return data._utils.collate.default_collate(new_batch)



# Modified custom collate function with dynamic size support (size changes every N batches) using Tensor resize (should be a bit faster)
def custom_collate_resize_tensor_fn(batch):
    import torch.nn.functional as F
    import time

    # start_time = time.time()

    if config.dynamic_size and config.dynamic_size_batch > 0:
        # Initialize persistent variables if they don't exist
        if not hasattr(custom_collate_resize_fn, 'batch_counter'):
            custom_collate_resize_fn.batch_counter = 0
            custom_collate_resize_fn.current_size = None

        if custom_collate_resize_fn.batch_counter % config.dynamic_size_batch == 0 or custom_collate_resize_fn.current_size is None:
            # Pick new random width and height
            chosen_width = random.choice(config.dynamic_size)
            chosen_height = random.choice(config.dynamic_size)
            custom_collate_resize_fn.current_size = (chosen_width, chosen_height)
            # print(f"[Dynamic Size] Changing to new size: {custom_collate_resize_fn.current_size} at batch {custom_collate_resize_fn.batch_counter}")

        custom_collate_resize_fn.batch_counter += 1
        data_size = custom_collate_resize_fn.current_size
    else:
        data_size = config.size

    new_batch = []

    # First convert PIL to tensors, then resize as tensors
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    for image, label, class_label, path in batch:
        # Convert PIL to tensor first
        image_tensor = to_tensor(image)
        label_tensor = to_tensor(label)

        # Resize tensors using interpolate (much faster than PIL resize)
        # data_size is (width, height), but interpolate expects (height, width)
        target_size = data_size[::-1]

        # Add batch dimension for interpolate, then remove it
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        label_tensor = F.interpolate(
            label_tensor.unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Apply normalization to image
        image_tensor = normalize(image_tensor)

        new_batch.append((image_tensor, label_tensor, class_label, path))

    result = data._utils.collate.default_collate(new_batch)

    end_time = time.time()
    # print(f"Collate function took: {end_time - start_time:.4f} seconds for size {data_size}")

    return result
