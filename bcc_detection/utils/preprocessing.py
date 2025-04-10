import os
import numpy as np
from PIL import Image
import openslide
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
import torch
from torchvision import transforms

def load_wsi(wsi_path):
    """Load a whole slide image using OpenSlide"""
    return openslide.OpenSlide(wsi_path)

def get_tissue_mask(slide, level=0):
    """Extract tissue mask from WSI"""
    # Get thumbnail at specified level
    thumbnail = slide.get_thumbnail(slide.level_dimensions[level])
    thumbnail = np.array(thumbnail)
    
    # Convert to grayscale
    if len(thumbnail.shape) == 3:
        gray = np.mean(thumbnail, axis=2)
    else:
        gray = thumbnail
        
    # Apply Otsu's thresholding
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    
    # Remove small objects
    binary = remove_small_objects(binary, min_size=500)
    
    return binary

def extract_tiles(slide, tissue_mask, tile_size, overlap=0):
    """Extract tiles from WSI based on tissue mask"""
    tiles = []
    coords = []
    
    # Calculate step size based on overlap
    step = int(tile_size * (1 - overlap))
    
    # Get dimensions
    width, height = slide.dimensions
    
    # Extract tiles
    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            # Check if tile contains tissue
            tile_mask = tissue_mask[y:y+tile_size, x:x+tile_size]
            if np.sum(tile_mask) / (tile_size * tile_size) > 0.5:
                tile = slide.read_region((x, y), 0, (tile_size, tile_size))
                tile = tile.convert('RGB')
                tiles.append(tile)
                coords.append((x, y))
                
    return tiles, coords

def preprocess_tile(tile, input_size):
    """Preprocess a single tile for model input"""
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(tile)

def create_tile_dataset(slide_path, config):
    """Create a dataset of tiles from a WSI"""
    # Load WSI
    slide = load_wsi(slide_path)
    
    # Get tissue mask
    tissue_mask = get_tissue_mask(slide)
    
    # Extract tiles
    tiles, coords = extract_tiles(
        slide,
        tissue_mask,
        config["tile_size"],
        config["tile_overlap"]
    )
    
    # Preprocess tiles
    processed_tiles = []
    for tile in tiles:
        processed_tile = preprocess_tile(tile, config["input_size"])
        processed_tiles.append(processed_tile)
        
    return torch.stack(processed_tiles), coords 