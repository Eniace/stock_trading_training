import mean_reverting
if __name__ =="__main__":
    obj = mean_reverting.mean_reverting()
    if obj.parameters['run_mod']==0:
        obj.mean_reverting()