# %%

"""Same as log_run_val but passing values individually.
"""
import sqlite3
import uuid

# %%


def log_run(model, dataset, features, accuracy, precision, recall, roc_auc, f1,  params, notes, db, project, analyst):
    uuid_val = str(uuid.uuid4())
    sql = f"INSERT INTO runs (ID,model,dataset,features,accuracy, precision, recall, roc_auc, f1, params, notes,project, analyst) VALUES('{uuid_val}','{model}','{dataset}','{features}',{accuracy},{precision},{recall},{roc_auc},{f1},'{params}','{notes}','{project}','{analyst}')"
    print(sql)
    dbase = sqlite3.connect(db)
    dbase.execute(sql)

    dbase.commit()

    print('\n>>> Record inserted')

    dbase.close()
    print('\n>>> Database Closed')

    return

# %%
