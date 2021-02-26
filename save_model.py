
from pandas import DataFrame

def save_model(model, history, modelname, time, memory):
    filename = 'Models/' + modelname                #Models are stored in a folder
    model.save(filename)                            #save the model
    df = DataFrame.from_dict(history)
    df['Model'] = modelname
    df['Time'] = time
    df['Memory'] = memory
    df.index.name = 'Epoch'
    df.index += 1 
    df.to_csv(filename + '/results' + modelname + '.csv' ,index=True) 