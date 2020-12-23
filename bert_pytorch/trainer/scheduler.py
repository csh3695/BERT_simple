

def linear_lr_scheduler(start_val=1e-7, warm_step=20, warm_val=3e-5, end=200, i=1):
    lr = lambda x: \
        start_val + (x-1)*(warm_val - start_val)/(warm_step - 1) if x <= warm_step \
        else (warm_step-x)*(warm_val)/(end-warm_step) + warm_val
    while True:
        yield lr(i)
        i += 1
