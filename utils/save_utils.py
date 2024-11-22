import torch
import pickle

def save_agent(agent,path):
    torch.save(agent,open(path,'wb'))
    cpu_agent=torch.load(open(path,'rb'), map_location=torch.device('cpu'))
    pickle.dump(cpu_agent,open(path,'wb'))

def load_agent(path):
    agent=pickle.load(open(path,'rb'))
    return agent


