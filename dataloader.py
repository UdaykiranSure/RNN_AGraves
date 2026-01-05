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
      """Parse a XML file from the IAM online dataset, returns a list of strokes """    
      tree = ET.parse(xml_path)   
      root = tree.getroot()    
      strokes = [ [(int(point.attrib["x"]), int(point.attrib["y"])) for point in stroke] for stroke in root[-1]]    
      return [np.asarray(stroke) for stroke in strokes]


def create_sequence(xml_path):
    """ TODO : should return the sequence as a numpy array
    create sequences by extracting strokes from xml file containing pen strokes"""
    strokes = parse_strokes(xml_path)
    offsets = [np.diff(stroke ,axis=0,prepend=stroke[:1]) for stroke in strokes]
    sequence = []
    for stroke in offsets:
        sequence.extend(np.concat([stroke, np.zeros((len(stroke),1))],axis = 1))
        sequence[-1][-1] = 1
    return sequence

def extract_sequences(data_dir):   
     text_paths = list_directories(data_dir)
     sequences = []
     for text_path in text_paths:
          sequence = create_sequence(text_path)
          sequences.append(sequence)
     return sequences

def collate_fn(batch):
     max_len = 200
     sequences = []
     lengths = []
     lens = [len(seq) for seq in batch]
    #  print('lens before trunc:', lens)
     for seq in batch:
          seq = torch.from_numpy(seq, dtype=torch.float32)
          seq = seq[:max_len]
          lengths.append(len(seq))
          sequences.append(seq)
     padded = pad_sequence(sequences, batch_first= True)
    #  print(lengths)
    
     return padded,lengths

def get_dataloader(data_dir,batch_size):
    sequences = extract_sequences(data_dir)
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
