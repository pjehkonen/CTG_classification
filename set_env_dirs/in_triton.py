import os

def in_triton():
    if os.path.exists('/scratch'):
        return True
    else:
        return False