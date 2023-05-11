from celery import Celery
import stackbraf

app = Celery('tasks', backend='rpc://' , broker='pyamqp://guest@localhost:5672//')

@app.task
def stackbraf_model(smiles, name):
    return stackbraf.execute_algorithm(smiles, name)