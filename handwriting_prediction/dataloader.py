import xml.etree.ElementTree as ET
import os
import numpy as np
from torch.utils.data import DataLoader 
from torch.nn.utils.rnn import pad_sequence  
import torch

def list_directories(root_dir):
     """Extract all xml paths from the line stokes folder of IAM ON-Line DB"""
     paths = []
     try:
         author_dirs = os.listdir(root_dir)
     except:
          print('root dir not found')
     for author in author_dirs:
          try:
               text_dirs = os.listdir(os.path.join(root_dir,author))
          except:
               print('text dirs not found',author)
          for text in text_dirs:
               try:
                   line_dirs = os.listdir(os.path.join(root_dir,author, text))
               except:
                    print('line dirs not found',author,line)
               for line in line_dirs:
                   xml_path = os.path.join(root_dir,author, text,line)
                   if line[-3:] == 'xml' and os.path.exists(xml_path):
                       paths.append(xml_path)
     return paths


def parse_strokes(xml_path: str):
    """
    Parse IAM XML file and return list of strokes
    Each stroke is a list of (x, y) after normalization
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- compute normalization offsets (same as target) ---
    x_offset = 1e20
    y_offset = 1e20
    y_height = 0

    for i in range(1, 4):
        x = float(root[0][i].attrib['x'])
        y = float(root[0][i].attrib['y'])
        x_offset = min(x_offset, x)
        y_offset = min(y_offset, y)
        y_height = max(y_height, y)

    y_height -= y_offset
    x_offset -= 100
    y_offset -= 100

    # --- parse strokes ---
    strokes = []
    for stroke in root[1].findall('Stroke'):
        points = []
        for point in stroke.findall('Point'):
            x = float(point.attrib['x']) - x_offset
            y = float(point.attrib['y']) - y_offset
            points.append((int(x), int(y)))
        strokes.append(np.asarray(points, dtype=np.int16))

    return strokes


def create_sequence(xml_path):
    """
    Convert strokes into (dx, dy, eos) sequence
    """
    strokes = parse_strokes(xml_path)

    total_points = sum(len(s) for s in strokes)
    sequence = np.zeros((total_points, 3), dtype=np.int16)

    prev_x, prev_y = 0, 0
    idx = 0

    for stroke in strokes:
        for i, (x, y) in enumerate(stroke):
            sequence[idx, 0] = x - prev_x
            sequence[idx, 1] = y - prev_y
            prev_x, prev_y = x, y

            if i == len(stroke) - 1:
                sequence[idx, 2] = 1  # end-of-stroke flag

            idx += 1

    return sequence


def extract_sequences(data_dir):
    xml_paths = list_directories(data_dir)
    sequences = []

    for xml_path in xml_paths:
        if xml_path.endswith(".xml"):
            print("processing", xml_path)
            sequences.append(create_sequence(xml_path))

    return sequences

def collate_fn(batch):
     max_len = 200
     sequences = []
     lengths = []
     lens = [len(seq) for seq in batch]
    #  print('lens before trunc:', lens)
     for seq in batch:
          seq = seq[:max_len]
          seq = np.asarray(seq,dtype = np.float32)
          seq = torch.tensor(seq, dtype=torch.float32)
          lengths.append(len(seq))
          sequences.append(seq)
     padded = pad_sequence(sequences, batch_first= True)
    #  print(lengths)
    
     return padded,lengths

def get_dataloader(data_dir,batch_size, split = 'train', val_ratio = 0.2):
     sequences = extract_sequences(data_dir)
     split_idx = int(len(sequences )* (1- val_ratio))
     if split == 'train':
          sequences = sequences[:split_idx]
     elif split == 'val':
          sequences = sequences[split_idx: ]

     dataloader = DataLoader(sequences,batch_size, collate_fn=collate_fn)
     return dataloader





# data_dir = '/Users/udaykiran/ML/ResearchPapers/rnn_graves/data/lineStrokes'
# batch_size = 32
# tr_dataloader = get_dataloader(data_dir, batch_size)
# for batch, lengths in tr_dataloader:
#      print(batch.shape, len(lengths))
#      break

# sequences = extract_sequences(data_dir)
# for seq in sequences:
#      pass






# xml_path = 'data/lineStrokes/a01/a01-000/a01-000u-01.xml'
# strokes = parse_strokes(xml_path)
# offsets = [np.diff(stroke ,axis=0,prepend=stroke[:1]) for stroke in strokes]
# len(offsets)
# offsets[0].shape
# seq = create_sequence(xml_path)
# type(seq)
# type(seq[0])
# len(seq)
