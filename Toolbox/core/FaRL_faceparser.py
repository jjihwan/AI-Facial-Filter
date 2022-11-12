import sys
import torch
sys.path.append('..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import facer

face_parser = facer.face_parser('farl/lapa/448', device=device)
face_detector = facer.face_detector('retinaface/mobilenet', device=device)

def FaRL(image):
  with torch.inference_mode():
    faces = face_detector(image)
    faces = face_parser(image, faces)
  seg_logits = faces['seg']['logits']
  seg_probs = seg_logits.softmax(dim=1)
  output = torch.zeros(1,4,image.shape[2],image.shape[3])
  output[0,0] = seg_probs[0,1]
  output[0,1] = seg_probs[0,4]+seg_probs[0,5]
  output[0,2] = seg_probs[0,8]+seg_probs[0,9]
  output[0,3] = seg_probs[0,6]
  
  output = round(output,0)

  return output
