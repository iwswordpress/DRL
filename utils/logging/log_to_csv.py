
from csv import writer


def save_results(file, data):
    with open(file, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(data)
        f_object.close()

    print('end csv logger function save_results()')

# %%
