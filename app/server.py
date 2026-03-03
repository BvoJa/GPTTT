from fastapi import FastAPI
import numpy as np
import onnxruntime as ort
import socket

def generate_onnx(session, idx, max_new_tokens, block_size):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        
        outputs = session.run([output_name], {input_name: idx_cond})
        logits = outputs[0]
        
        logits = logits[:, -1, :] 
        
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        batch_size = idx.shape[0]
        vocab_size = probs.shape[1]
        idx_next = np.empty((batch_size, 1), dtype=np.int64)
        
        for i in range(batch_size):
            idx_next[i, 0] = np.random.choice(vocab_size, p=probs[i])
            
        idx = np.concatenate((idx, idx_next), axis=1)
        
    return idx


onnx_path = "app/model.onnx"
    
session = ort.InferenceSession(onnx_path)

with open('app/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

block_size = 256 

app = FastAPI()

@app.get('/')
def read_root():
    return {
        'message': 'GPT model API',
        'worker': socket.gethostname()
    }

@app.post('/predict')
def predict(data: dict):
    # prompt = "O Romeo, Romeo! wherefore "
    context = [stoi[c] for c in data["prompt"] if c in stoi]
    idx = np.array([context], dtype=np.int64) 
    
    generated_idx = generate_onnx(
        session=session, 
        idx=idx, 
        max_new_tokens=100,  
        block_size=block_size
    )
    
    generated_text = "".join([itos[int(i)] for i in generated_idx[0]])

    return {
        'sentence' : generated_text,
        'worker': socket.gethostname()
    }